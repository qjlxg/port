import pandas as pd
import requests
import numpy as np
import json
import re
from datetime import datetime, timedelta
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import warnings

# 忽略 DeprecationWarning 警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 筛选条件（放宽）
MIN_RETURN = 3.0  # 年化收益率 ≥ 3%
MAX_VOLATILITY = 25.0  # 波动率 ≤ 25%
MIN_SHARPE = 0.2  # 夏普比率 ≥ 0.2
MAX_FEE = 2.5  # 管理费 ≤ 2.5%
RISK_FREE_RATE = 3.0  # 无风险利率 3%
MIN_DAYS = 100  # 最低数据天数（从 252 放宽到 100）
TIMEOUT = 15  # 网络请求超时时间（秒）

# 基金类型筛选，可选：'全部'，'混合型'，'股票型'，'指数型'，'债券型'，'QDII'，'FOF'
FUND_TYPE_FILTER = '全部'

# 配置 requests 重试机制
session = requests.Session()
retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# 随机 User-Agent 列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

# 申万行业分类数据（简化版，仅作演示）
# 实际项目中，您需要从可靠的数据源动态获取
SW_INDUSTRY_MAPPING = {
    '600519': '食品饮料',  # 贵州茅台
    '000858': '食品饮料',  # 五粮液
    '002475': '家用电器',  # 立讯精密
    '002415': '家用电器',  # 海康威视
    '300750': '计算机',    # 宁德时代
    '300059': '传媒',      # 东方财富
    '002460': '汽车',      # 赣锋锂业
    '600036': '金融',      # 招商银行
    '600276': '医药生物',  # 恒瑞医药
    '600030': '金融',      # 中信证券
}

# 步骤 1: 动态获取全市场基金列表
def get_all_funds_from_eastmoney():
    """
    动态从天天基金网获取全市场基金列表。
    """
    print(">>> 步骤1: 正在动态获取全市场基金列表...")
    url = "http://fund.eastmoney.com/js/fundcode_search.js"
    headers = {'User-Agent': random.choice(USER_AGENTS), 'Referer': 'http://fund.eastmoney.com/'}
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        content = response.text
        match = re.search(r'var\s+r\s*=\s*(\[.*?\]);', content, re.DOTALL)
        if match:
            fund_data = json.loads(match.group(1))
            df = pd.DataFrame(fund_data, columns=['code', 'pinyin', 'name', 'type', 'pinyin_full'])
            df = df[['code', 'name', 'type']].drop_duplicates(subset=['code'])
            print(f"    √ 成功获取到 {len(df)} 只基金。")
            return df
        print("    × 未能解析基金列表数据。")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"    × 获取基金列表失败: {e}")
        return pd.DataFrame()

# 步骤 2: 获取历史净值（主备双重保险）
def get_fund_net_values(code, start_date, end_date):
    """
    尝试从两个接口获取基金历史净值数据。
    """
    # 尝试主接口 (pingzhongdata)
    df, latest_value = get_net_values_from_pingzhongdata(code, start_date, end_date)
    if not df.empty and len(df) >= MIN_DAYS:
        return df, latest_value, 'pingzhongdata'
    # 如果主接口失败，尝试备用接口 (lsjz)
    df, latest_value = get_net_values_from_lsjz(code, start_date, end_date)
    if not df.empty and len(df) >= MIN_DAYS:
        return df, latest_value, 'lsjz'
    return pd.DataFrame(), None, 'None'

def get_net_values_from_pingzhongdata(code, start_date, end_date):
    """从 fund.eastmoney.com/pingzhongdata/ 接口获取净值"""
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {'User-Agent': random.choice(USER_AGENTS), 'Referer': f'http://fund.eastmoney.com/{code}.html'}
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        net_worth_match = re.search(r'Data_netWorthTrend\s*=\s*(\[.*?\]);', response.text, re.DOTALL)
        if not net_worth_match: return pd.DataFrame(), None
        net_worth_list = json.loads(net_worth_match.group(1))
        df = pd.DataFrame(net_worth_list).rename(columns={'x': 'date', 'y': 'net_value'})
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)
        latest_value = df['net_value'].iloc[-1] if not df.empty else None
        return df, latest_value
    except requests.exceptions.RequestException: return pd.DataFrame(), None
    except (json.JSONDecodeError, IndexError): return pd.DataFrame(), None

def get_net_values_from_lsjz(code, start_date, end_date):
    """从 fund.eastmoney.com/f10/lsjz 接口获取净值"""
    url = f"http://fund.eastmoney.com/f10/lsjz?fundCode={code}&pageIndex=1&pageSize=50000"
    headers = {'User-Agent': random.choice(USER_AGENTS), 'Referer': f'http://fund.eastmoney.com/f10/fjcc_{code}.html'}
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        data_str_match = re.search(r'var\s+apidata=\{content:"(.*?)",', response.text, re.DOTALL)
        if not data_str_match: return pd.DataFrame(), None
        json_data_str = data_str_match.group(1).replace("\\", "")
        data = json.loads(json_data_str)
        if 'LSJZList' not in data or not data['LSJZList']: return pd.DataFrame(), None
        df = pd.DataFrame(data['LSJZList']).rename(columns={'FSRQ': 'date', 'DWJZ': 'net_value'})
        df['date'] = pd.to_datetime(df['date'])
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)
        latest_value = df['net_value'].iloc[-1] if not df.empty else None
        return df, latest_value
    except requests.exceptions.RequestException: return pd.DataFrame(), None
    except (json.JSONDecodeError, IndexError): return pd.DataFrame(), None

# 步骤 3: 获取实时估值
def get_fund_realtime_estimate(code):
    """从 fundgz.1234567.com.cn 接口获取基金实时估值。"""
    url = f"http://fundgz.1234567.com.cn/js/{code}.js?rt={int(time.time() * 1000)}"
    headers = {'User-Agent': random.choice(USER_AGENTS), 'Referer': f'http://fund.eastmoney.com/{code}.html'}
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        match = re.search(r'jsonpgz\((.*)\)', response.text, re.DOTALL)
        if match:
            json_data = json.loads(match.group(1))
            gsz = json_data.get('gsz')
            return float(gsz) if gsz else None
        return None
    except Exception:
        return None

# 步骤 4: 获取管理费
def get_fund_fee(code):
    """从 fund.eastmoney.com/pingzhongdata/ 接口获取管理费。"""
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {'User-Agent': random.choice(USER_AGENTS), 'Referer': f'http://fund.eastmoney.com/{code}.html'}
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        fee_match = re.search(r'data_fundTribble\.ManagerFee=\'([\d.]+)\'', response.text)
        if fee_match:
            fee = float(fee_match.group(1))
            return fee
        return 1.5
    except requests.exceptions.RequestException:
        return 1.5

# 步骤 5: 获取基金最新持仓
def get_fund_holdings(code):
    """从天天基金网获取基金最新持仓信息。"""
    url = f"http://fund.eastmoney.com/DataCenter/Fund/JJZCHoldDetail.aspx?fundCode={code}"
    headers = {'User-Agent': random.choice(USER_AGENTS), 'Referer': f'http://fund.eastmoney.com/f10/jjcc_{code}.html'}
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        holdings = []
        stock_table = soup.find('table', {'class': 'm-table'})
        if stock_table:
            for row in stock_table.find_all('tr')[1:]:
                cells = row.find_all('td')
                if len(cells) >= 4:
                    holdings.append({
                        'name': cells[1].text.strip(),
                        'code': cells[2].text.strip(),
                        'ratio': cells[3].text.strip()
                    })
        return holdings
    except requests.exceptions.RequestException: return []
    except Exception: return []

# 步骤 6: 计算指标
def calculate_metrics(net_df, start_date, end_date):
    """计算基金的年化收益率、波动率和夏普比率。"""
    net_df = net_df[(net_df['date'] >= pd.to_datetime(start_date)) & (net_df['date'] <= pd.to_datetime(end_date))].copy()
    if len(net_df) < MIN_DAYS: return None
    returns = net_df['net_value'].pct_change().dropna()
    total_return = (net_df['net_value'].iloc[-1] / net_df['net_value'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_return *= 100
    volatility = returns.std() * np.sqrt(252) * 100
    sharpe = (annual_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
    return {
        'annual_return': round(annual_return, 2),
        'volatility': round(volatility, 2),
        'sharpe': round(sharpe, 2)
    }

# 步骤 7: 增强持仓分析 - 行业分布与集中度
def analyze_holdings(holdings):
    """
    分析基金持仓的行业分布和集中度。
    """
    industry_counts = {}
    industry_ratios = {}
    
    for holding in holdings:
        stock_code = holding.get('code', 'N/A')
        ratio_str = holding.get('ratio', '0%').replace('%', '')
        ratio = float(ratio_str) if ratio_str else 0
        
        # 根据股票代码查找行业
        industry = SW_INDUSTRY_MAPPING.get(stock_code, '其他')
        
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        industry_ratios[industry] = industry_ratios.get(industry, 0) + ratio
    
    # 转换为DataFrame进行排序
    industry_df = pd.DataFrame(list(industry_ratios.items()), columns=['行业', '占比 (%)'])
    industry_df = industry_df.sort_values(by='占比 (%)', ascending=False)
    
    # 计算行业集中度（前三大行业占比）
    top3_concentration = industry_df['占比 (%)'].iloc[:3].sum() if len(industry_df) >= 3 else industry_df['占比 (%)'].sum()
    
    return industry_df, round(top3_concentration, 2)


# 主函数
def main():
    start_time = time.time()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime('%Y-%m-%d')

    funds_df = get_all_funds_from_eastmoney()
    if funds_df.empty:
        print("无法获取基金列表，程序退出。")
        return

    # 基金类型筛选
    if FUND_TYPE_FILTER != '全部':
        funds_df = funds_df[funds_df['type'] == FUND_TYPE_FILTER].copy()
        print(f">>> 步骤2: 已根据您的偏好筛选，保留 {len(funds_df)} 只{FUND_TYPE_FILTER}基金。")
    
    print(">>> 步骤3: 正在逐一处理基金数据并进行筛选...")
    results = []
    
    # 预先获取市场指数数据
    index_code = '000300' # 沪深300指数
    index_df = get_net_values_from_pingzhongdata(index_code, start_date, end_date)[0]
    if index_df.empty:
        print(f"    × 无法获取市场指数 {index_code} 的历史数据，将无法计算贝塔系数。")

    for idx, row in funds_df.iterrows():
        code = row['code']
        name = row['name']
        
        print(f"\n    正在处理基金 {idx+1}/{len(funds_df)}: {name} ({code})...")

        # 获取历史净值数据
        net_df, latest_net_value, data_source = get_fund_net_values(code, start_date, end_date)
        
        if net_df.empty:
            print(f"    跳过：净值数据不足。")
            continue

        # 计算指标
        metrics = calculate_metrics(net_df, start_date, end_date)
        if metrics is None:
            print(f"    跳过：数据不足 {MIN_DAYS} 天（仅有 {len(net_df)} 天）。")
            continue

        # 获取管理费和实时估值
        fee = get_fund_fee(code)
        realtime_estimate = get_fund_realtime_estimate(code)

        # 筛选
        if (metrics['annual_return'] >= MIN_RETURN and
            metrics['volatility'] <= MAX_VOLATILITY and
            metrics['sharpe'] >= MIN_SHARPE and
            fee <= MAX_FEE):
            
            result = {
                '基金代码': code,
                '基金名称': name,
                '年化收益率 (%)': metrics['annual_return'],
                '年化波动率 (%)': metrics['volatility'],
                '夏普比率': metrics['sharpe'],
                '管理费 (%)': round(fee, 2),
                '最新净值': latest_net_value,
                '实时估值': round(realtime_estimate, 4) if realtime_estimate is not None else 'N/A'
            }
            score = (0.6 * (metrics['annual_return'] / 20) + 0.3 * metrics['sharpe'] + 0.1 * (2 - fee))
            result['综合评分'] = round(score, 2)
            results.append(result)
            print(f"    √ 通过筛选，评分: {result['综合评分']:.2f}")
        else:
            print("    × 未通过筛选。")

    # 输出结果并获取持仓信息
    if results:
        final_df = pd.DataFrame(results).sort_values('综合评分', ascending=False)
        final_df = final_df.reset_index(drop=True)
        final_df.index = final_df.index + 1
        
        print("\n--- 步骤4: 筛选完成，开始输出推荐基金列表 ---")
        print(final_df)
        final_df.to_csv('recommended_cn_funds.csv', index=True, index_label='排名', encoding='utf-8-sig')
        print("\n>>> 推荐结果已保存至 recommended_cn_funds.csv 文件。")

        for idx, row in final_df.iterrows():
            code = row['基金代码']
            name = row['基金名称']
            
            print(f"\n--- 正在分析基金 {name} ({code}) 的持仓详情 ---")
            
            holdings = get_fund_holdings(code)
            
            if holdings:
                print(f"    √ 成功获取到最新十大持仓。")
                print("      - 持仓股票:")
                for holding in holdings:
                    print(f"        股票名称: {holding.get('name', 'N/A')}, 股票代码: {holding.get('code', 'N/A')}, 占比: {holding.get('ratio', 'N/A')}")
                
                industry_df, concentration = analyze_holdings(holdings)
                print(f"\n      - 行业分析:")
                print(industry_df.to_string(index=False))
                print(f"        行业集中度（前三大行业占比）: {concentration:.2f}%")
            else:
                print(f"    × 未能获取到最新持仓数据。")
    else:
        print("\n没有找到符合条件的基金，建议调整筛选条件。")
        
    end_time = time.time()
    print(f"\n>>> 整个程序执行完毕，耗时: {(end_time - start_time):.2f} 秒。")

if __name__ == "__main__":
    main()
