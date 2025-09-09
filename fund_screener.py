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

# 基金类型筛选，可选：'全部'，'混合型'，'股票型'，'指数型'，'债券型'，'QDII'，'FOF'
# 请根据您的偏好选择，如果选择'全部'则不进行类型筛选
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

# 步骤 1: 动态获取全市场基金列表
def get_all_funds_from_eastmoney():
    """
    动态从天天基金网获取全市场基金列表。
    """
    print("正在动态获取全市场基金列表...")
    url = "http://fund.eastmoney.com/js/fundcode_search.js"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': 'http://fund.eastmoney.com/'
    }
    try:
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        content = response.text
        match = re.search(r'var\s+r\s*=\s*(\[.*?\]);', content, re.DOTALL)
        if match:
            fund_data = json.loads(match.group(1))
            df = pd.DataFrame(fund_data, columns=['code', 'pinyin', 'name', 'type', 'pinyin_full'])
            df = df[['code', 'name', 'type']].drop_duplicates(subset=['code'])
            print(f"成功获取到 {len(df)} 只基金。")
            return df
        print("未能解析基金列表数据。")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"获取基金列表失败: {e}")
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
    print(f"主接口获取失败，尝试备用接口 (lsjz)...")
    df, latest_value = get_net_values_from_lsjz(code, start_date, end_date)
    if not df.empty and len(df) >= MIN_DAYS:
        return df, latest_value, 'lsjz'

    # 如果两个接口都失败，返回空
    print(f"基金 {code} 数据获取失败")
    return pd.DataFrame(), None, 'None'

def get_net_values_from_pingzhongdata(code, start_date, end_date):
    """从 fund.eastmoney.com/pingzhongdata/ 接口获取净值"""
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Host': 'fund.eastmoney.com',
        'Accept': 'application/javascript, */*;q=0.8'
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        if not response.text.strip():
            return pd.DataFrame(), None

        net_worth_match = re.search(r'Data_netWorthTrend\s*=\s*(\[.*?\]);', response.text, re.DOTALL)
        if not net_worth_match:
            return pd.DataFrame(), None

        try:
            net_worth_list = json.loads(net_worth_match.group(1))
        except json.JSONDecodeError:
            return pd.DataFrame(), None

        df = pd.DataFrame(net_worth_list)
        if 'x' not in df or 'y' not in df:
            return pd.DataFrame(), None

        df = df.rename(columns={'x': 'date', 'y': 'net_value'})
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df[(df['date'] >= pd.to_datetime(start_date)) &
                (df['date'] <= pd.to_datetime(end_date))]
        df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)

        if df.empty:
            return pd.DataFrame(), None

        latest_value = df['net_value'].iloc[-1] if not df.empty else None
        return df, latest_value

    except requests.exceptions.RequestException:
        return pd.DataFrame(), None

def get_net_values_from_lsjz(code, start_date, end_date):
    """从 fund.eastmoney.com/f10/lsjz 接口获取净值"""
    url = f"http://fund.eastmoney.com/f10/lsjz?fundCode={code}&pageIndex=1&pageSize=50000"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/f10/fjcc_{code}.html',
        'Host': 'fund.eastmoney.com',
        'Accept': 'application/json, text/javascript, */*; q=0.01'
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        data_str_match = re.search(r'var\s+apidata=\{content:"(.*?)",', response.text, re.DOTALL)
        if not data_str_match:
            return pd.DataFrame(), None

        json_data_str = data_str_match.group(1).replace("\\", "")
        try:
            data = json.loads(json_data_str)
        except json.JSONDecodeError:
            return pd.DataFrame(), None

        if 'LSJZList' not in data or not data['LSJZList']:
            return pd.DataFrame(), None

        df = pd.DataFrame(data['LSJZList'])
        df = df.rename(columns={'FSRQ': 'date', 'DWJZ': 'net_value', 'LJJZ': 'total_value', 'JZZZL': 'daily_return'})
        df['date'] = pd.to_datetime(df['date'])
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df[(df['date'] >= pd.to_datetime(start_date)) &
                (df['date'] <= pd.to_datetime(end_date))]
        df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)

        if df.empty:
            return pd.DataFrame(), None

        latest_value = df['net_value'].iloc[-1] if not df.empty else None
        return df, latest_value

    except requests.exceptions.RequestException:
        return pd.DataFrame(), None

# 新增函数: 获取市场指数历史净值
def get_index_net_values(code, start_date, end_date):
    """
    获取市场指数的历史净值数据。
    """
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Host': 'fund.eastmoney.com',
        'Accept': 'application/javascript, */*;q=0.8'
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        net_worth_match = re.search(r'Data_netWorthTrend\s*=\s*(\[.*?\]);', response.text, re.DOTALL)
        if not net_worth_match:
            return pd.DataFrame()

        net_worth_list = json.loads(net_worth_match.group(1))
        df = pd.DataFrame(net_worth_list)
        df = df.rename(columns={'x': 'date', 'y': 'net_value'})
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df[(df['date'] >= pd.to_datetime(start_date)) &
                (df['date'] <= pd.to_datetime(end_date))]
        df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

# 步骤 3: 获取实时估值
def get_fund_realtime_estimate(code):
    """
    从 fundgz.1234567.com.cn 接口获取基金实时估值。
    """
    url = f"http://fundgz.1234567.com.cn/js/{code}.js?rt={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Host': 'fundgz.1234567.com.cn'
    }
    try:
        response = session.get(url, headers=headers, timeout=10)
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
    """
    从 fund.eastmoney.com/pingzhongdata/ 接口获取管理费。
    """
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Host': 'fund.eastmoney.com'
    }
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        fee_match = re.search(r'data_fundTribble\.ManagerFee=\'([\d.]+)\'', response.text)
        if fee_match:
            fee = float(fee_match.group(1))
            return fee
        return 1.5
    except requests.exceptions.RequestException:
        return 1.5

# 步骤 5: 获取基金持仓信息
def get_fund_holdings(code):
    """
    从天天基金网获取基金最新持仓信息和更新日期。
    """
    url = f"http://fund.eastmoney.com/DataCenter/Fund/JJZCHoldDetail.aspx?fundCode={code}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/f10/jjcc_{code}.html'
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        holdings = []
        # 查找持仓表格
        stock_table = soup.find('table', {'class': 'm-table'})
        
        # 获取更新日期
        update_date = 'N/A'
        date_match = re.search(r'截至(.+?)季报', response.text)
        if date_match:
            update_date = date_match.group(1).strip()
        
        if stock_table:
            for row in stock_table.find_all('tr')[1:]: # 跳过表头
                cells = row.find_all('td')
                if len(cells) >= 4:
                    holdings.append({
                        'name': cells[1].text.strip(),
                        'code': cells[2].text.strip(),
                        'ratio': cells[3].text.strip()
                    })
        return holdings, update_date
    except requests.exceptions.RequestException:
        return [], 'N/A'
    except Exception:
        return [], 'N/A'

# 新增函数: 获取基金历史持仓信息
def get_fund_historical_holdings(code):
    """
    从天天基金网获取基金历史持仓信息。
    """
    url = f"https://fundf10.eastmoney.com/ccmx_{code}.html"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'https://fundf10.eastmoney.com/ccmx_{code}.html'
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        historical_data = {}
        # 查找所有季度报告的表格
        report_sections = soup.find_all('div', class_='boxitem')
        
        for section in report_sections:
            title_tag = section.find('h4', class_='title')
            if not title_tag:
                continue
            
            report_date = title_tag.text.strip().replace('基金持仓', '').replace('前十持仓', '').strip()
            
            holdings = []
            table = section.find('table')
            if table:
                for row in table.find_all('tr')[1:]:
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        holdings.append({
                            'name': cells[1].text.strip(),
                            'code': cells[2].text.strip(),
                            'ratio': cells[3].text.strip()
                        })
            
            if holdings:
                historical_data[report_date] = holdings
        return historical_data
        
    except requests.exceptions.RequestException:
        return {}
    except Exception:
        return {}

# 新增函数: 计算贝塔系数
def calculate_beta(fund_df, index_df):
    """
    计算基金的贝塔系数。
    """
    if fund_df.empty or index_df.empty:
        return None

    # 合并基金和指数数据，按日期对齐
    merged_df = pd.merge(fund_df, index_df, on='date', suffixes=('_fund', '_index'))

    if len(merged_df) < 2:
        return None

    # 计算日收益率
    merged_df['return_fund'] = merged_df['net_value_fund'].pct_change()
    merged_df['return_index'] = merged_df['net_value_index'].pct_change()
    merged_df = merged_df.dropna()

    if len(merged_df) < 2:
        return None
        
    # 计算协方差和方差
    cov_matrix = np.cov(merged_df['return_fund'], merged_df['return_index'])
    covariance = cov_matrix[0, 1]
    variance_market = cov_matrix[1, 1]

    if variance_market == 0:
        return None

    beta = covariance / variance_market
    return round(beta, 2)


# 步骤 6: 计算指标
def calculate_metrics(net_df, start_date, end_date):
    """
    计算基金的年化收益率、波动率和夏普比率。
    """
    net_df = net_df[(net_df['date'] >= pd.to_datetime(start_date)) &
                    (net_df['date'] <= pd.to_datetime(end_date))].copy()
    
    if len(net_df) < MIN_DAYS:
        return None
        
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

# 主函数
def main():
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime('%Y-%m-%d')  # 缩短到 3 年

    funds_df = get_all_funds_from_eastmoney()
    if funds_df.empty:
        print("无法获取基金列表，程序退出。")
        return

    # 基金类型筛选
    if FUND_TYPE_FILTER != '全部':
        funds_df = funds_df[funds_df['type'] == FUND_TYPE_FILTER].copy()
        print(f"已根据您的偏好筛选，保留 {len(funds_df)} 只{FUND_TYPE_FILTER}基金。")

    print(f"获取到 {len(funds_df)} 只候选基金。")
    results = []
    debug_data = []

    # 预先获取市场指数数据
    index_code = '000300' # 沪深300指数
    index_df = get_index_net_values(index_code, start_date, end_date)
    if index_df.empty:
        print(f"无法获取市场指数 {index_code} 的历史数据，将无法计算贝塔系数。")

    for idx, row in funds_df.iterrows():
        code = row['code']
        name = row['name']
        
        # 优化打印，每处理10只基金打印一次，避免日志过长
        if (idx + 1) % 10 == 0:
             print(f"\n正在处理第 {idx+1}/{len(funds_df)} 只基金: {name} ({code})")

        # 获取历史净值数据
        net_df, latest_net_value, data_source = get_fund_net_values(code, start_date, end_date)
        
        if net_df.empty:
            debug_data.append({
                '基金代码': code,
                '基金名称': name,
                '失败原因': '无净值数据',
                '数据来源': data_source
            })
            continue

        # 计算指标
        metrics = calculate_metrics(net_df, start_date, end_date)
        if metrics is None:
            debug_data.append({
                '基金代码': code,
                '基金名称': name,
                '失败原因': f'数据不足 {MIN_DAYS} 天（仅有 {len(net_df)} 天）',
                '数据来源': data_source,
                '数据条数': len(net_df)
            })
            continue

        # 获取管理费
        fee = get_fund_fee(code)
        
        # 获取实时估值
        realtime_estimate = get_fund_realtime_estimate(code)

        # 计算贝塔系数
        beta = None
        if not index_df.empty:
            beta = calculate_beta(net_df, index_df)
        
        debug_info = {
            '基金代码': code,
            '基金名称': name,
            '年化收益率 (%)': metrics['annual_return'],
            '年化波动率 (%)': metrics['volatility'],
            '夏普比率': metrics['sharpe'],
            '管理费 (%)': fee,
            '最新净值': latest_net_value,
            '实时估值': realtime_estimate,
            '贝塔系数': beta,
            '数据来源': data_source,
            '数据条数': len(net_df),
            '数据开始日期': net_df['date'].iloc[0].strftime('%Y-%m-%d') if not net_df.empty else None,
            '数据结束日期': net_df['date'].iloc[-1].strftime('%Y-%m-%d') if not net_df.empty else None
        }

        # 筛选
        if (metrics['annual_return'] >= MIN_RETURN and
            metrics['volatility'] <= MAX_VOLATILITY and
            metrics['sharpe'] >= MIN_SHARPE and
            fee <= MAX_FEE):
            
            # 情景分析
            if beta is not None:
                bull_market_return = beta * 10
                bear_market_return = beta * -10
                flat_market_return = beta * 0
            else:
                bull_market_return, bear_market_return, flat_market_return = 'N/A', 'N/A', 'N/A'

            result = {
                '基金代码': code,
                '基金名称': name,
                '年化收益率 (%)': metrics['annual_return'],
                '年化波动率 (%)': metrics['volatility'],
                '夏普比率': metrics['sharpe'],
                '管理费 (%)': round(fee, 2),
                '最新净值': latest_net_value,
                '实时估值': round(realtime_estimate, 4) if realtime_estimate is not None else 'N/A',
                '贝塔系数': beta,
                '牛市预期收益 (%)': round(bull_market_return, 2) if isinstance(bull_market_return, (int, float)) else bull_market_return,
                '熊市预期收益 (%)': round(bear_market_return, 2) if isinstance(bear_market_return, (int, float)) else bear_market_return,
                '数据来源': data_source
            }
            score = (0.6 * (metrics['annual_return'] / 20) +
                     0.3 * metrics['sharpe'] +
                     0.1 * (2 - fee))
            result['综合评分'] = round(score, 2)
            results.append(result)
        else:
            debug_info['失败原因'] = '未通过筛选'
        debug_data.append(debug_info)
        time.sleep(random.uniform(1, 3))  # 随机延时 1-3 秒

    # 保存调试信息
    debug_df = pd.DataFrame(debug_data)
    debug_df.to_csv('debug_fund_metrics.csv', index=False, encoding='utf-8-sig')
    print("\n调试信息已保存至 debug_fund_metrics.csv")

    # 输出结果并获取持仓信息
    if results:
        final_df = pd.DataFrame(results).sort_values('综合评分', ascending=False)
        print("\n--- 符合条件的推荐基金列表 ---")
        print(final_df)
        final_df.to_csv('recommended_cn_funds.csv', index=False, encoding='utf-8-sig')
        print("\n结果已保存至 recommended_cn_funds.csv 文件。")

        for idx, row in final_df.iterrows():
            code = row['基金代码']
            name = row['基金名称']
            
            print(f"\n--- 正在获取基金 {name} ({code}) 的持仓详情 ---")
            
            # 获取最新持仓
            latest_holdings, update_date = get_fund_holdings(code)
            is_outdated = 'N/A'
            if update_date != 'N/A':
                try:
                    date_obj = datetime.strptime(update_date, '%Y-%m-%d')
                    if (datetime.now() - date_obj).days > 90:
                        is_outdated = '可能过时'
                    else:
                        is_outdated = '最新'
                except ValueError:
                    is_outdated = '日期格式错误'
            
            if latest_holdings:
                print(f"    - 最新持仓更新日期: {update_date} ({is_outdated})")
                for holding in latest_holdings:
                    print(f"      - 股票名称: {holding.get('name', 'N/A')}, 占比: {holding.get('ratio', 'N/A')}")
            else:
                print(f"    - 未能获取到最新持仓数据。")
            
            # 获取历史持仓
            historical_data = get_fund_historical_holdings(code)
            if historical_data:
                print(f"    - 历史持仓数据:")
                for date, holdings_list in historical_data.items():
                    print(f"        - {date} 持仓:")
                    for holding in holdings_list:
                         print(f"          - 股票名称: {holding.get('name', 'N/A')}, 占比: {holding.get('ratio', 'N/A')}")
            else:
                print(f"    - 未能获取到历史持仓数据。")
            
            print("-" * 50)
            time.sleep(random.uniform(2, 5)) # 随机延时 2-5 秒
            
    else:
        print("\n没有找到符合条件的基金，建议调整筛选条件。")

if __name__ == "__main__":
    main()
