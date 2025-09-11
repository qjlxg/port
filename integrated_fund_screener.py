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
import traceback
from playwright.sync_api import sync_playwright

# 忽略警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 筛选条件
MIN_RETURN = 3.0  # 年化收益率 ≥ 3%
MAX_VOLATILITY = 25.0  # 波动率 ≤ 25%
MIN_SHARPE = 0.2  # 夏普比率 ≥ 0.2
MAX_FEE = 2.5  # 管理费 ≤ 2.5%
RISK_FREE_RATE = 3.0  # 无风险利率 3%
MIN_DAYS = 100  # 最低数据天数
TIMEOUT = 10  # 网络请求超时时间（秒）
FUND_TYPE_FILTER = ['混合型', '股票型']  # 基金类型筛选，这里为了示例只保留混合型和股票型

# 配置 requests 重试机制
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# 随机 User-Agent 和 Headers
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

# 申万行业分类数据
SW_INDUSTRY_MAPPING = {
    '600519': '食品饮料', '000858': '食品饮料', '002475': '家用电器', '002415': '家用电器',
    '300750': '计算机', '300059': '传媒', '002460': '汽车', '600036': '金融',
    '600276': '医药生物', '600030': '金融', '000001': '金融', '600000': '金融',
    '601318': '金融', '601166': '金融', '000333': '家用电器', '000651': '家用电器',
    '600690': '家用电器', '002304': '食品饮料', '000568': '食品饮料', '600809': '食品饮料',
    '603288': '食品饮料', '300760': '医药生物', '002714': '农林牧渔', '601012': '电力设备',
    '300274': '电力设备', '601688': '金融', '600837': '金融', '601398': '金融',
    '601288': '金融', '002241': '计算机', '300033': '计算机', '002594': '汽车',
    '601633': '汽车', '603259': '医药生物', '300122': '医药生物', '600196': '医药生物',
    '000423': '医药生物', '002007': '医药生物', '600085': '医药生物', '600660': '汽车',
    '002920': '计算机', '300628': '计算机', '600893': '电力设备', '300014': '电力设备',
    '601985': '电力设备', '002027': '传媒', '300027': '传媒', '002739': '传媒',
    '000725': '电子', '300223': '电子', '600584': '电子', '600887': '食品饮料',
    '603888': '食品饮料'
}

# 数据缓存目录
CACHE_DIR = "fund_data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_all_funds_from_eastmoney():
    """从东方财富网获取所有基金列表"""
    print(">>> 步骤1: 正在动态获取全市场基金列表...", flush=True)
    cache_file = os.path.join(CACHE_DIR, "fund_list.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                funds_df = pickle.load(f)
            print(f"    √ 从缓存加载 {len(funds_df)} 只基金。", flush=True)
            return funds_df
        except Exception as e:
            print(f"    × 加载基金列表缓存失败: {e}，将重新获取。", flush=True)

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
            df = pd.DataFrame(fund_data, columns=['基金代码', '拼音', '名称', '类型', '全拼'])
            df = df[['基金代码', '名称', '类型']].drop_duplicates(subset=['基金代码'])
            df = df[df['类型'].isin(FUND_TYPE_FILTER)].copy()
            df = df.reset_index(drop=True)
            print(f"    √ 获取到 {len(df)} 只{', '.join(FUND_TYPE_FILTER)}基金。", flush=True)
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
            return df
        print("    × 未能解析基金列表数据。", flush=True)
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"    × 获取基金列表失败: {e}", flush=True)
        return pd.DataFrame()
    except Exception as e:
        print(f"    × 解析基金列表时发生异常: {e}", flush=True)
        return pd.DataFrame()

def get_fund_rankings(start_date, end_date, fund_type):
    """
    获取指定时间段和类型的基金排名数据。
    返回一个包含基金代码和排名的 DataFrame。
    """
    print(f"    正在获取 {fund_type} 类型基金在 {start_date} 至 {end_date} 期间的排名...", flush=True)
    url = f"http://fund.eastmoney.com/data/rankhandler.aspx?op=dy&dt=kf&ft={fund_type}&rs=&gs=0&sc=qjzf&st=desc&sd={start_date}&ed={end_date}&es=1&qdii=&pi=1&pn=10000&dx=1"
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        match = re.search(r'var\s+rankData\s*=\s*(\[.*?\]);', response.text)
        if not match:
            print(f"    × 未找到 {fund_type} 类型在 {start_date} 至 {end_date} 期间的排名数据。")
            return pd.DataFrame()

        data_list = json.loads(match.group(1))
        df = pd.DataFrame(data_list, columns=['基金代码', '名称', '日期', '单位净值', '累计净值', '日增长率',
                                        '近1周', '近1月', '近3月', '近6月', '近1年', '近2年',
                                        '近3年', '今年来', '成立来', '成立日期', '手续费'])
        
        # 仅保留基金代码和名称
        df = df[['基金代码', '名称']].copy()
        
        print(f"    √ 成功获取 {len(df)} 条数据。", flush=True)
        return df
    except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError) as e:
        print(f"    × 获取排名数据失败: {e}", flush=True)
        return pd.DataFrame()

def comprehensive_screener():
    """
    执行基金的综合筛选流程。
    包括四四三三法则筛选和量化指标筛选。
    """
    all_funds_df = get_all_funds_from_eastmoney()
    if all_funds_df.empty:
        return pd.DataFrame()

    print(">>> 步骤2: 执行 '四四三三法则' 筛选...", flush=True)
    
    # 获取不同时间段的排名数据
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    start_3y_str = (datetime.now() - timedelta(days=3 * 365)).strftime('%Y-%m-%d')
    start_2y_str = (datetime.now() - timedelta(days=2 * 365)).strftime('%Y-%m-%d')
    start_1y_str = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    start_6m_str = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    start_3m_str = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # 统一使用混合型基金排名数据进行筛选
    df_3y = get_fund_rankings(start_3y_str, end_date_str, 'hh')
    df_2y = get_fund_rankings(start_2y_str, end_date_str, 'hh')
    df_1y = get_fund_rankings(start_1y_str, end_date_str, 'hh')
    df_6m = get_fund_rankings(start_6m_str, end_date_str, 'hh')
    df_3m = get_fund_rankings(start_3m_str, end_date_str, 'hh')
    
    # 获取排名前 25% 的基金列表
    top25_3y = df_3y.head(int(len(df_3y) * 0.25))
    top25_2y = df_2y.head(int(len(df_2y) * 0.25))
    top25_1y = df_1y.head(int(len(df_1y) * 0.25))
    
    # 获取排名前 33.3% 的基金列表
    top33_6m = df_6m.head(int(len(df_6m) * 0.333))
    top33_3m = df_3m.head(int(len(df_3m) * 0.333))
    
    # 执行多维度合并筛选
    # 使用 merge 代替 join，并指定 on='基金代码' 和 suffixes，以解决列名冲突
    # 这里通过 on='基金代码' 来合并，并为重复的'名称'列添加后缀
    selected_funds_df = pd.merge(top25_3y, top25_2y, on='基金代码', how='inner', suffixes=('_3y', '_2y'))
    selected_funds_df = pd.merge(selected_funds_df, top25_1y, on='基金代码', how='inner', suffixes=('', '_1y'))
    selected_funds_df = pd.merge(selected_funds_df, top33_6m, on='基金代码', how='inner', suffixes=('', '_6m'))
    selected_funds_df = pd.merge(selected_funds_df, top33_3m, on='基金代码', how='inner', suffixes=('', '_3m'))
    
    # 清理掉多余的名称列，只保留一个
    selected_funds_df = selected_funds_df.loc[:, ~selected_funds_df.columns.duplicated()]
    
    print(f"    √ '四四三三法则' 筛选完成，剩余 {len(selected_funds_df)} 只基金。", flush=True)

    if selected_funds_df.empty:
        print("    × 无基金通过 '四四三三法则' 筛选，无法进行下一步分析。", flush=True)
        return pd.DataFrame()

    selected_funds_df = pd.merge(selected_funds_df, all_funds_df[['基金代码', '名称', '类型']], on='基金代码', how='left')
    selected_funds_df.rename(columns={'名称_x': '名称'}, inplace=True)
    selected_funds_df = selected_funds_df.drop(columns=['名称_y'], errors='ignore')
    
    # 保存初步筛选结果
    selected_funds_df.to_csv('selected_funds.csv', index=False, encoding='utf-8-sig')
    print(f"    初步筛选结果已保存至 selected_funds.csv", flush=True)

    return selected_funds_df
    
def get_fund_net_values(code, start_date, end_date):
    """
    获取基金历史净值数据
    """
    df, latest_value = get_net_values_from_pingzhongdata(code, start_date, end_date)
    if not df.empty and len(df) >= MIN_DAYS:
        return df, latest_value, 'pingzhongdata'
    return pd.DataFrame(), None, 'None'

def get_net_values_from_pingzhongdata(code, start_date, end_date):
    """从 fund.eastmoney.com/pingzhongdata 接口获取净值"""
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Accept': 'text/javascript, application/javascript, */*',
        'Connection': 'keep-alive'
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
    except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError) as e:
        return pd.DataFrame(), None

def get_fund_fee(code):
    """获取基金管理费率"""
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        fee_match = re.search(r'data_fundTribble\.ManagerFee=\'([\d.]+)\'', response.text)
        fee = float(fee_match.group(1)) if fee_match else 1.5
        return fee
    except Exception:
        return 1.5

def get_fund_holdings(code):
    """使用 Playwright 获取基金持仓数据"""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"http://fundf10.eastmoney.com/ccmx_{code}.html", timeout=60000)
            page.wait_for_selector('div.boxitem table', timeout=60000)
            html_content = page.content()
            browser.close()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            stock_table = soup.find('div', class_='boxitem').find('table')
            
            if stock_table:
                holdings = []
                for row in stock_table.find_all('tr')[1:]:
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        holdings.append({
                            'name': cells[1].text.strip(),
                            'code': cells[2].text.strip(),
                            'ratio': cells[3].text.strip().replace('%', '')
                        })
                return holdings
            return []
    except Exception as e:
        print(f"    × Playwright 请求或解析失败: {e}", flush=True)
        return []

def calculate_metrics(net_df):
    """计算基金量化指标"""
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

def process_fund(row, end_date):
    """处理单只基金，获取详细信息并进行量化筛选"""
    code = row['基金代码']
    name = row['名称']
    fund_type = row['类型']

    print(f"\n--- 正在深度分析基金: {name} ({code})...", flush=True)

    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime('%Y-%m-%d')
    net_df, latest_net_value, _ = get_fund_net_values(code, start_date, end_date)
    
    if net_df.empty or len(net_df) < MIN_DAYS:
        print(f"    × 基金 {name} ({code}) 数据不足，跳过。", flush=True)
        return None

    metrics = calculate_metrics(net_df)
    if metrics is None:
        print(f"    × 基金 {name} ({code}) 指标计算失败，跳过。", flush=True)
        return None

    fee = get_fund_fee(code)
    holdings = get_fund_holdings(code)
    
    is_passed = (metrics['annual_return'] >= MIN_RETURN and
                 metrics['volatility'] <= MAX_VOLATILITY and
                 metrics['sharpe'] >= MIN_SHARPE and
                 fee <= MAX_FEE)

    if not is_passed:
        print(f"    × 基金 {name} ({code}) 未通过量化指标筛选，跳过。", flush=True)
        return None
    
    # 将持仓数据转换为可读格式
    holdings_str = ", ".join([f"{h['name']}({h['ratio']}%)" for h in holdings]) if holdings else "无数据"

    result = {
        '基金代码': code,
        '基金名称': name,
        '基金类型': fund_type,
        '年化收益率 (%)': metrics['annual_return'],
        '年化波动率 (%)': metrics['volatility'],
        '夏普比率': metrics['sharpe'],
        '最大回撤 (%)': 'N/A', # 该脚本中未计算，可根据需要补充
        '管理费 (%)': round(fee, 2),
        '前十大持仓': holdings_str
    }
    
    print(f"    √ 基金 {name} ({code}) 通过所有筛选。", flush=True)
    return result

def main():
    print("--- 启动基金综合筛选器 ---", flush=True)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 步骤1: 执行初步筛选 (四四三三法则)
    selected_funds_df = comprehensive_screener()
    if selected_funds_df.empty:
        print("\n>>> 未找到符合初步筛选条件的基金。", flush=True)
        return

    # 步骤2: 对初步筛选结果进行深度分析和量化筛选
    print("\n>>> 步骤3: 正在对初步筛选结果进行深度分析和量化筛选...", flush=True)
    final_results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_fund, row.to_dict(), end_date) for _, row in selected_funds_df.iterrows()]
        
        for future in tqdm(futures, desc="深度分析中", total=len(selected_funds_df)):
            try:
                result = future.result()
                if result:
                    final_results.append(result)
            except Exception as e:
                print(f"    × 处理基金时发生异常: {e}", flush=True)
                traceback.print_exc()

    # 步骤3: 保存最终结果
    if final_results:
        final_df = pd.DataFrame(final_results)
        final_df.sort_values('年化收益率 (%)', ascending=False, inplace=True)
        final_df.to_csv('final_detailed_screener_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n>>> 最终筛选结果已保存至 final_detailed_screener_results.csv，共 {len(final_df)} 只基金。", flush=True)
        print("\n--- 最终推荐基金列表 ---", flush=True)
        print(final_df.to_string(), flush=True)
    else:
        print("\n>>> 未找到符合所有条件的基金，建议调整筛选条件。", flush=True)

    print(f"\n--- 筛选器运行完毕 ---", flush=True)

if __name__ == "__main__":
    main()
