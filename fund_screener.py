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
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import pickle
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 筛选条件（与原始代码一致）
MIN_RETURN = 3.0  # 年化收益率 ≥ 3%
MAX_VOLATILITY = 25.0  # 波动率 ≤ 25%
MIN_SHARPE = 0.2  # 夏普比率 ≥ 0.2
MAX_FEE = 2.5  # 管理费 ≤ 2.5%
RISK_FREE_RATE = 3.0  # 无风险利率 3%
MIN_DAYS = 100  # 最低数据天数
TIMEOUT = 15  # 网络请求超时时间（秒，延长以减少超时）
FUND_TYPE_FILTER = ['混合型', '股票型', '指数型']  # 基金类型筛选

# 配置 requests 重试机制
session = requests.Session()
retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# 随机 User-Agent 和 Headers
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

# 扩展的申万行业分类数据
SW_INDUSTRY_MAPPING = {
    '600519': '食品饮料', '000858': '食品饮料', '002475': '家用电器', '002415': '家用电器',
    '300750': '计算机', '300059': '传媒', '002460': '汽车', '600036': '金融',
    '600276': '医药生物', '600030': '金融',
    '000001': '金融', '600000': '金融', '601318': '金融', '601166': '金融',
    '000333': '家用电器', '000651': '家用电器', '600690': '家用电器',
    '002304': '食品饮料', '000568': '食品饮料', '600809': '食品饮料', '603288': '食品饮料',
    '300760': '医药生物', '002714': '农林牧渔', '601012': '电力设备', '300274': '电力设备',
    '601688': '金融', '600837': '金融', '601398': '金融', '601288': '金融',
    '002241': '计算机', '300033': '计算机', '002594': '汽车', '601633': '汽车',
    '603259': '医药生物', '300122': '医药生物', '600196': '医药生物',
    '000423': '医药生物', '002007': '医药生物', '600085': '医药生物',
    '600660': '汽车', '002920': '计算机', '300628': '计算机',
    '600893': '电力设备', '300014': '电力设备', '601985': '电力设备',
    '002027': '传媒', '300027': '传媒', '002739': '传媒',
    '000725': '电子', '300223': '电子', '600584': '电子',
    '600887': '食品饮料', '603888': '食品饮料'
}

# 数据缓存目录
CACHE_DIR = "fund_data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 步骤 1: 获取全市场基金列表
def get_all_funds_from_eastmoney():
    cache_file = os.path.join(CACHE_DIR, "fund_list.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            funds_df = pickle.load(f)
        print(f"    √ 从缓存加载 {len(funds_df)} 只基金。", flush=True)
        return funds_df

    print(">>> 步骤1: 正在动态获取全市场基金列表...", flush=True)
    url = "http://fund.eastmoney.com/js/fundcode_search.js"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': 'http://fund.eastmoney.com/',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Connection': 'keep-alive'
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        content = response.text
        match = re.search(r'var\s+r\s*=\s*(\[.*?\]);', content, re.DOTALL)
        if match:
            fund_data = json.loads(match.group(1))
            df = pd.DataFrame(fund_data, columns=['code', 'pinyin', 'name', 'type', 'pinyin_full'])
            df = df[['code', 'name', 'type']].drop_duplicates(subset=['code'])
            df = df[df['type'].isin(FUND_TYPE_FILTER)].copy()
            print(f"    √ 获取到 {len(df)} 只{', '.join(FUND_TYPE_FILTER)}基金。", flush=True)
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
            return df
        print("    × 未能解析基金列表数据。", flush=True)
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"    × 获取基金列表失败: {e}", flush=True)
        return pd.DataFrame()

# 步骤 2: 获取历史净值
def get_fund_net_values(code, start_date, end_date):
    cache_file = os.path.join(CACHE_DIR, f"net_values_{code}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            net_df, latest_value = pickle.load(f)
        if not net_df.empty and len(net_df) >= MIN_DAYS:
            return net_df, latest_value, 'cache'

    df, latest_value = get_net_values_from_pingzhongdata(code, start_date, end_date)
    if not df.empty and len(df) >= MIN_DAYS:
        with open(cache_file, "wb") as f:
            pickle.dump((df, latest_value), f)
        return df, latest_value, 'pingzhongdata'

    df, latest_value = get_net_values_from_lsjz(code, start_date, end_date)
    if not df.empty and len(df) >= MIN_DAYS:
        with open(cache_file, "wb") as f:
            pickle.dump((df, latest_value), f)
        return df, latest_value, 'lsjz'
    return pd.DataFrame(), None, 'None'

def get_net_values_from_pingzhongdata(code, start_date, end_date):
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Accept': 'text/javascript, application/javascript, */*'
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        net_worth_match = re.search(r'Data_netWorthTrend\s*=\s*(\[.*?\]);', response.text, re.DOTALL)
        if not net_worth_match:
            return pd.DataFrame(), None
        net_worth_list = json.loads(net_worth_match.group(1))
        df = pd.DataFrame(net_worth_list).rename(columns={'x': 'date', 'y': 'net_value'})
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)
        latest_value = df['net_value'].iloc[-1] if not df.empty else None
        return df, latest_value
    except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError):
        return pd.DataFrame(), None

def get_net_values_from_lsjz(code, start_date, end_date):
    url = f"http://fund.eastmoney.com/f10/lsjz?fundCode={code}&pageIndex=1&pageSize=50000"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/f10/jjfl_{code}.html'
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        data_str_match = re.search(r'var\s+apidata=\{content:"(.*?)",', response.text, re.DOTALL)
        if not data_str_match:
            return pd.DataFrame(), None
        json_data_str = data_str_match.group(1).replace("\\", "")
        data = json.loads(json_data_str)
        if 'LSJZList' in data and data['LSJZList']:
            df = pd.DataFrame(data['LSJZList']).rename(columns={'FSRQ': 'date', 'DWJZ': 'net_value'})
            df['date'] = pd.to_datetime(df['date'])
            df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
            df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
            df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)
            latest_value = df['net_value'].iloc[-1] if not df.empty else None
            return df, latest_value
        return pd.DataFrame(), None
    except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError):
        return pd.DataFrame(), None

# 步骤 3: 获取实时估值
def get_fund_realtime_estimate(code):
    url = f"http://fundgz.1234567.com.cn/js/{code}.js?rt={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html'
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        match = re.search(r'jsonpgz\((.*)\)', response.text)
        if match:
            json_data = json.loads(match.group(1))
            gsz = json_data.get('gsz')
            return float(gsz) if gsz else None
        return None
    except Exception:
        return None

# 步骤 4: 获取管理费
def get_fund_fee(code):
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html'
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        fee_match = re.search(r'data_fundTribble\.ManagerFee=\'([\d.]+)\'', response.text)
        fee = float(fee_match.group(1)) if fee_match else 1.5
        return fee
    except requests.exceptions.RequestException:
        return 1.5

# 步骤 5: 获取基金最新持仓（修复版）
def get_fund_holdings(code):
    # 首先尝试从 pingzhongdata 接口获取
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Accept': 'text/javascript, application/javascript, */*'
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        holdings_match = re.search(r'Data_holdStock\s*=\s*(\[.*?\]);', response.text, re.DOTALL)
        if holdings_match:
            holdings_list = json.loads(holdings_match.group(1))
            holdings = []
            for item in holdings_list:
                holdings.append({
                    'name': item.get('stockName', 'N/A'),
                    'code': item.get('stockCode', 'N/A'),
                    'ratio': item.get('holdPercent', '0%')
                })
            print(f"    √ 从 pingzhongdata 获取到 {code} 的持仓数据。", flush=True)
            return holdings
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"    × pingzhongdata 接口获取持仓失败: {e}，尝试 FundShareHolding 接口。", flush=True)

    # 备用接口：从 FundShareHolding 接口获取
    url = f"http://fund.eastmoney.com/data/FundShareHolding.ashx?code={code}&top=10&rt={int(time.time() * 1000)}"
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        match = re.search(r'jsonFundShareHolding\((.*?)\)', response.text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            holdings_list = data.get('list', [])
            holdings = []
            for item in holdings_list:
                holdings.append({
                    'name': item.get('stockname', 'N/A'),
                    'code': item.get('stockcode', 'N/A'),
                    'ratio': str(item.get('percent', 0)) + '%'
                })
            if holdings:
                print(f"    √ 从 FundShareHolding 接口获取到 {code} 的持仓数据。", flush=True)
                return holdings
            print(f"    × FundShareHolding 接口未获取到有效持仓数据: {code}", flush=True)
            return []
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"    × FundShareHolding 接口获取持仓失败: {e}", flush=True)
        return []
    finally:
        time.sleep(random.uniform(1, 3))  # 随机延迟

# 计算指标
def calculate_metrics(net_df, start_date, end_date, index_df):
    if len(net_df) < MIN_DAYS:
        return None
    net_df = net_df[(net_df['date'] >= pd.to_datetime(start_date)) & (net_df['date'] <= pd.to_datetime(end_date))].copy()
    returns = net_df['net_value'].pct_change().dropna()
    total_return = (net_df['net_value'].iloc[-1] / net_df['net_value'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_return *= 100
    volatility = returns.std() * np.sqrt(252) * 100
    sharpe = (annual_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0

    # 计算最大回撤
    rolling_max = net_df['net_value'].expanding().max()
    drawdown = (net_df['net_value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    # 计算贝塔系数
    beta = None
    if not index_df.empty:
        index_df = index_df[(index_df['date'] >= pd.to_datetime(start_date)) & (index_df['date'] <= pd.to_datetime(end_date))].copy()
        index_returns = index_df['net_value'].pct_change().dropna()
        aligned_df = pd.merge(net_df[['date', 'net_value']], index_df[['date', 'net_value']],
                             on='date', how='inner', suffixes=('_fund', '_index'))
        if len(aligned_df) >= 2:
            fund_returns = aligned_df['net_value_fund'].pct_change().dropna()
            index_returns = aligned_df['net_value_index'].pct_change().dropna()
            if len(fund_returns) >= 2 and len(index_returns) >= 2:
                cov_matrix = np.cov(fund_returns, index_returns)
                beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else None

    return {
        'annual_return': round(annual_return, 2),
        'volatility': round(volatility, 2),
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(max_drawdown, 2),
        'beta': round(beta, 2) if beta is not None else None
    }

# 分析持仓行业分布
def analyze_holdings(holdings):
    if not holdings:
        return pd.DataFrame(), 0
    industry_counts = {}
    total_ratio = 0
    for holding in holdings:
        stock_code = holding.get('code', 'N/A')
        ratio = float(holding.get('ratio', '0').replace('%', '')) if isinstance(holding.get('ratio'), str) else holding.get('ratio', 0)
        industry = SW_INDUSTRY_MAPPING.get(stock_code, '其他')
        industry_counts[industry] = industry_counts.get(industry, 0) + ratio
        total_ratio += ratio
    industry_df = pd.DataFrame(list(industry_counts.items()), columns=['行业', '占比 (%)'])
    industry_df['占比 (%)'] = industry_df['占比 (%)'].apply(lambda x: round(x, 2))
    industry_df = industry_df.sort_values('占比 (%)', ascending=False).reset_index(drop=True)
    concentration = sum(industry_df['占比 (%)'].head(3)) if len(industry_df) >= 3 else sum(industry_df['占比 (%)'])
    return industry_df, round(concentration, 2)

# 处理单只基金
def process_fund(row, start_date, end_date, index_df, total_funds, idx):
    start_time = time.time()
    code, name, fund_type = row.code, row.name, row.type
    debug_info = {'基金代码': code, '基金名称': name, '基金类型': fund_type}
    print(f"\n--- 正在处理基金 {idx}/{total_funds}: {name} ({code})...", flush=True)

    try:
        net_df, latest_net_value, data_source = get_fund_net_values(code, start_date, end_date)
        if net_df.empty or len(net_df) < MIN_DAYS:
            debug_info['筛选状态'] = '未通过'
            debug_info['失败原因'] = f'数据天数不足 {MIN_DAYS}'
            print(f"    × 跳过：数据不足 {MIN_DAYS} 天（{data_source}）。", flush=True)
            return None, debug_info

        metrics = calculate_metrics(net_df, start_date, end_date, index_df)
        if metrics is None:
            debug_info['筛选状态'] = '未通过'
            debug_info['失败原因'] = f'数据天数不足 {MIN_DAYS}'
            print(f"    × 跳过：计算指标失败，数据不足。", flush=True)
            return None, debug_info

        fee = get_fund_fee(code)
        reasons = []
        realtime_estimate = get_fund_realtime_estimate(code)
        holdings = get_fund_holdings(code)
        industry_df, concentration = analyze_holdings(holdings) if holdings else (pd.DataFrame(), 0)

        is_passed = (metrics['annual_return'] >= MIN_RETURN and
                     metrics['volatility'] <= MAX_VOLATILITY and
                     metrics['sharpe'] >= MIN_SHARPE and
                     fee <= MAX_FEE)

        debug_info.update({
            '年化收益率 (%)': metrics['annual_return'],
            '年化波动率 (%)': metrics['volatility'],
            '夏普比率': metrics['sharpe'],
            '贝塔系数': metrics['beta'],
            '最大回撤 (%)': metrics['max_drawdown'],
            '管理费 (%)': round(fee, 2),
            '处理耗时': round(time.time() - start_time, 2)
        })

        if not is_passed:
            if metrics['annual_return'] < MIN_RETURN:
                reasons.append(f"年化收益率 ({metrics['annual_return']}%) < {MIN_RETURN}%")
            if metrics['volatility'] > MAX_VOLATILITY:
                reasons.append(f"波动率 ({metrics['volatility']}%) > {MAX_VOLATILITY}%")
            if metrics['sharpe'] < MIN_SHARPE:
                reasons.append(f"夏普比率 ({metrics['sharpe']}) < {MIN_SHARPE}")
            if fee > MAX_FEE:
                reasons.append(f"管理费 ({fee}%) > {MAX_FEE}%")
            debug_info['筛选状态'] = '未通过'
            debug_info['失败原因'] = ' / '.join(reasons)
            print(f"    × 未通过筛选。原因：{' / '.join(reasons)}", flush=True)
            return None, debug_info

        score = (0.6 * (metrics['annual_return'] / 20) + 0.3 * metrics['sharpe'] + 0.1 * (2 - fee))
        debug_info['筛选状态'] = '通过'
        debug_info['综合评分'] = round(score, 2)

        result = {
            '基金代码': code,
            '基金名称': name,
            '基金类型': fund_type,
            '年化收益率 (%)': metrics['annual_return'],
            '年化波动率 (%)': metrics['volatility'],
            '夏普比率': metrics['sharpe'],
            '贝塔系数': metrics['beta'],
            '最大回撤 (%)': metrics['max_drawdown'],
            '管理费 (%)': round(fee, 2),
            '最新净值': latest_net_value,
            '实时估值': round(realtime_estimate, 4) if realtime_estimate else 'N/A',
            '综合评分': round(score, 2),
            '行业分布': industry_df.to_dict('records') if not industry_df.empty else [],
            '行业集中度 (%)': concentration
        }
        print(f"    √ 通过筛选，评分: {result['综合评分']:.2f}", flush=True)
        return result, debug_info
    except Exception as e:
        debug_info['筛选状态'] = '未通过'
        debug_info['失败原因'] = f'异常错误: {str(e)}'
        print(f"    × 处理基金 {name} ({code}) 异常: {e}", flush=True)
        return None, debug_info

# 主函数
def main():
    print(">>> 基金筛选工具启动...", flush=True)
    start_time = time.time()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime('%Y-%m-%d')

    # 获取基金列表
    funds_df = get_all_funds_from_eastmoney()
    if funds_df.empty:
        print("无法获取基金列表，程序退出。", flush=True)
        return

    total_funds = len(funds_df)
    print(f">>> 共 {total_funds} 只基金待处理（{', '.join(FUND_TYPE_FILTER)}）。", flush=True)

    # 获取市场指数数据
    index_code = '000300'  # 沪深300
    index_df = pd.DataFrame()
    try:
        index_df, _ = get_net_values_from_pingzhongdata(index_code, start_date, end_date)
        if index_df.empty:
            print(f"    × 无法获取市场指数 {index_code} 数据，尝试备用指数。", flush=True)
            index_code_fallback = '000001'  # 上证指数
            index_df, _ = get_net_values_from_pingzhongdata(index_code_fallback, start_date, end_date)
            if index_df.empty:
                print(f"    × 无法获取市场指数 {index_code_fallback} 数据，贝塔系数将不可用。", flush=True)
    except Exception as e:
        print(f"    × 获取市场指数数据异常: {e}，贝塔系数将不可用。", flush=True)

    # 并行处理基金
    results = []
    debug_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_fund, row, start_date, end_date, index_df, total_funds, idx)
                   for idx, row in enumerate(funds_df.itertuples(index=False), 1)]
        for future in tqdm(futures, desc="处理基金", total=total_funds):
            try:
                result, debug_info = future.result()
                debug_data.append(debug_info)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"    × 处理基金时发生异常: {e}", flush=True)

    # 保存结果
    if results:
        final_df = pd.DataFrame(results).sort_values('综合评分', ascending=False).reset_index(drop=True)
        final_df.index = final_df.index + 1
        print("\n--- 步骤4: 筛选完成，推荐基金列表 ---", flush=True)
        print(final_df.drop(columns=['行业分布']), flush=True)
        final_df.to_csv('recommended_cn_funds.csv', index=True, index_label='排名', encoding='utf-8-sig')
        print("\n>>> 推荐结果已保存至 recommended_cn_funds.csv", flush=True)

        # 输出持仓详情
        for idx, row in final_df.iterrows():
            code = row['基金代码']
            name = row['基金名称']
            print(f"\n--- 基金 {name} ({code}) 持仓详情 ---", flush=True)
            if row['行业分布']:
                industry_df = pd.DataFrame(row['行业分布'])
                print(industry_df.to_string(index=False), flush=True)
                print(f"    行业集中度（前三大行业占比）: {row['行业集中度 (%)']:.2f}%", flush=True)
            else:
                print("    × 无持仓数据。", flush=True)
    else:
        print("\n>>> 未找到符合条件的基金，建议调整筛选条件。", flush=True)

    # 保存调试信息
    debug_df = pd.DataFrame(debug_data)
    debug_df.to_csv('debug_fund_metrics.csv', index=False, encoding='utf-8-sig')
    print(f">>> 调试信息已保存至 debug_fund_metrics.csv", flush=True)

    print(f">>> 总耗时: {round(time.time() - start_time, 2)}秒", flush=True)

if __name__ == "__main__":
    main()
