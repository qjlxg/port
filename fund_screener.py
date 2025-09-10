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
TIMEOUT = 10  # 网络请求超时时间（秒）
FUND_TYPE_FILTER = ['混合型', '股票型', '指数型']  # 基金类型筛选

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

# 扩展的申万行业分类数据（进一步扩展）
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

    # 新增的更可靠的接口
    url = 'https://api.fund.eastmoney.com/f10/FundNetWorthForMonth?fundCode={}&startDate={}&endDate={}&t={}'.format(
        code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), int(time.time() * 1000)
    )
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fundf10.eastmoney.com/jdzf_{code}.html',
        'Connection': 'keep-alive',
        'Accept': 'application/json, text/plain, */*',
    }
    try:
    	response = session.get(url, headers=headers, timeout=TIMEOUT)
    	response.raise_for_status()
    	json_data = response.json()
    	data = json_data.get('Data', {})
    	if data and data.get('fundNetWorthForMonth'):
    		fund_data = data['fundNetWorthForMonth']
    		df = pd.DataFrame(fund_data)
    		df['date'] = pd.to_datetime(df['JZRQ'])
    		df['value'] = pd.to_numeric(df['DWJZ'])
    		df = df[['date', 'value']].set_index('date').sort_index()
    		df = df.loc[start_date:end_date].copy()
    		latest_value = pd.to_numeric(data['fundDetails'].get('DWJZ'))
    		
    		if not df.empty and len(df) >= MIN_DAYS:
    			with open(cache_file, "wb") as f:
    				pickle.dump((df, latest_value), f)
    			return df, latest_value, 'fetch'
    		else:
    			print(f"    调试: {code} 新接口数据点不足 ({len(df)}天 < {MIN_DAYS}天)。", flush=True)
    			return pd.DataFrame(), None, 'fail'
    except Exception as e:
    	print(f"    调试: 新接口 {url} 请求或解析失败: {e}", flush=True)

    # 原始的 pingzhongdata 接口作为备用
    def get_net_values_from_pingzhongdata(code, start_date, end_date):
        url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Referer': f'http://fund.eastmoney.com/{code}.html',
        }
        try:
            response = session.get(url, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            match = re.search(r'var\s+Data_netWorthTrend\s*=\s*(\[.*?\]);.*var\s+Data_grandTotalNetWorth\s*=\s*(\[.*?\]);', response.text, re.DOTALL)
            if match:
                worth_data = json.loads(match.group(1))
                worth_total_data = json.loads(match.group(2))
                df = pd.DataFrame(worth_data, columns=['timestamp', 'value', 'daily_change', 'label'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
                df['value'] = pd.to_numeric(df['value'])
                df = df[['date', 'value']].set_index('date').sort_index()
                df = df.loc[start_date:end_date].copy()
                latest_value = worth_total_data[-1][1] if worth_total_data else None
                return df, latest_value
            print(f"    调试: http://fund.eastmoney.com/pingzhongdata/{code}.js JSON 解析失败，尝试下一个接口。", flush=True)
        except Exception as e:
            print(f"    调试: http://fund.eastmoney.com/pingzhongdata/{code}.js 解析异常: {e}", flush=True)
        return pd.DataFrame(), None

    df, latest_value = get_net_values_from_pingzhongdata(code, start_date, end_date)
    if not df.empty and len(df) >= MIN_DAYS:
        with open(cache_file, "wb") as f:
            pickle.dump((df, latest_value), f)
        return df, latest_value, 'fetch'
    
    return pd.DataFrame(), None, 'fail'

# 步骤 3: 获取估值
def get_fund_gsz(code):
    cache_file = os.path.join(CACHE_DIR, f"gsz_{code}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    url = f"http://fundgz.fund.eastmoney.com/api/FundGZ.ashx?t={int(time.time() * 1000)}&gzid={code}&rt=1&callback=jsonpgz"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Accept': 'text/javascript, application/javascript, */*',
        'Connection': 'keep-alive'
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        match = re.search(r'jsonpgz\((.*)\)', response.text, re.DOTALL)
        if match:
            json_data = json.loads(match.group(1))
            gsz = json_data.get('gsz')
            if gsz:
                with open(cache_file, "wb") as f:
                    pickle.dump(float(gsz), f)
                return float(gsz)
        return None
    except Exception:
        return None

# 步骤 4: 获取管理费
def get_fund_fee(code):
    cache_file = os.path.join(CACHE_DIR, f"fee_{code}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
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
        fee_match = re.search(r'var\s+fund_sourceRate\s*=\s*"(.*?)";', response.text)
        if fee_match:
            fee = float(fee_match.group(1))
            with open(cache_file, "wb") as f:
                pickle.dump(fee, f)
            return fee
    except Exception:
        pass
    return None

# 步骤 5: 获取股票持仓并计算行业分布
def get_fund_stock_positions(code):
    cache_file = os.path.join(CACHE_DIR, f"holdings_{code}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # 优先尝试新的、更稳定的接口
    url = f"https://api.fund.eastmoney.com/f10/FundHoldShares?fundCode={code}&year=&quarter=&t={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fundf10.eastmoney.com/ccmx_{code}.html',
        'Connection': 'keep-alive',
        'Accept': 'application/json, text/plain, */*',
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        json_data = response.json()
        data = json_data.get('Data', {})
        holdings = []
        if data and data.get('fundStocks'):
            for stock in data['fundStocks']:
                holdings.append({
                    'name': stock.get('GPJC'),
                    'code': stock.get('GPDM'),
                    'ratio': stock.get('JZBL')
                })
            
            # 计算行业分布和集中度
            if holdings:
                industry_mapping = SW_INDUSTRY_MAPPING
                industry_holdings = {}
                total_ratio = 0
                for item in holdings:
                    stock_code = item['code']
                    ratio = float(item['ratio'])
                    if stock_code in industry_mapping:
                        industry = industry_mapping[stock_code]
                        industry_holdings[industry] = industry_holdings.get(industry, 0) + ratio
                        total_ratio += ratio

                # 排序并计算集中度
                sorted_industries = sorted(industry_holdings.items(), key=lambda x: x[1], reverse=True)
                top_3_concentration = sum([ratio for _, ratio in sorted_industries[:3]])
                
                result = {
                    '持仓股票': holdings,
                    '行业分布': [{'行业': industry, '占比(%)': ratio} for industry, ratio in sorted_industries],
                    '行业集中度 (%)': top_3_concentration
                }

                print(f"    √ 获取 {code} 持仓数据成功，通过新API接口。", flush=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
                return result
    except Exception as e:
        print(f"    调试: {url} 请求或解析失败: {e}", flush=True)

    print(f"    × {code} 所有持仓数据接口均失败，无持仓数据。", flush=True)
    return {'持仓股票': [], '行业分布': [], '行业集中度 (%)': 0}

# 步骤 6: 计算贝塔系数
def calculate_beta(fund_returns, index_returns):
    common_dates = fund_returns.index.intersection(index_returns.index)
    if len(common_dates) < 2:
        return 0, 0
    
    fund_returns_aligned = fund_returns.loc[common_dates]
    index_returns_aligned = index_returns.loc[common_dates]
    
    # 确保数据为数值类型
    fund_returns_aligned = pd.to_numeric(fund_returns_aligned, errors='coerce').dropna()
    index_returns_aligned = pd.to_numeric(index_returns_aligned, errors='coerce').dropna()
    
    if len(fund_returns_aligned) < 2 or len(index_returns_aligned) < 2:
        return 0, 0
    
    # 计算协方差和指数方差
    cov_matrix = np.cov(fund_returns_aligned, index_returns_aligned)
    if cov_matrix.shape == (2, 2):
        covariance = cov_matrix[0, 1]
        index_variance = cov_matrix[1, 1]
    else:
        # 当只有单个数据点时np.cov返回一维数组
        return 0, 0

    if index_variance == 0:
        return 0, 0
    
    beta = covariance / index_variance
    alpha = (fund_returns_aligned.mean() - index_returns_aligned.mean() * beta) * 252
    
    return beta, alpha

# 步骤 7: 核心筛选逻辑
def process_fund(row, start_date, end_date, index_df, total_funds, idx):
    code = row.code
    name = row.name
    fund_type = row.type
    reasons = []
    debug_info = {'基金代码': code, '基金名称': name, '基金类型': fund_type}
    
    print(f"\n--- 正在处理基金 {idx}/{total_funds} ({', '.join(FUND_TYPE_FILTER)}): {name} ({code})...", flush=True)

    start_time = time.time()
    net_df, latest_net_value, data_source = get_fund_net_values(code, start_date, end_date)
    debug_info['数据源'] = data_source
    debug_info['数据点数'] = len(net_df) if not net_df.empty else 0

    if net_df.empty or len(net_df) < MIN_DAYS:
        reasons.append(f"数据不足（{len(net_df)}天 < {MIN_DAYS}天）")
        debug_info['筛选状态'] = '未通过'
        print(f"    × 未通过筛选。原因：{reasons[0]}", flush=True)
        return None

    # 计算指标
    daily_returns = net_df['value'].pct_change().dropna()
    if daily_returns.empty:
        reasons.append("无法计算日收益率")
        debug_info['筛选状态'] = '未通过'
        print(f"    × 未通过筛选。原因：{reasons[0]}", flush=True)
        return None

    annual_return = daily_returns.mean() * 252 * 100
    volatility = daily_returns.std() * np.sqrt(252) * 100
    sharpe_ratio = (annual_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
    
    # 获取管理费
    fund_fee = get_fund_fee(code)

    # 筛选条件判断
    if annual_return < MIN_RETURN:
        reasons.append(f"年化收益率 ({annual_return:.2f}%) < {MIN_RETURN}%")
    if volatility > MAX_VOLATILITY:
        reasons.append(f"波动率 ({volatility:.2f}%) > {MAX_VOLATILITY}%")
    if sharpe_ratio < MIN_SHARPE:
        reasons.append(f"夏普比率 ({sharpe_ratio:.2f}) < {MIN_SHARPE}")
    if fund_fee is not None and fund_fee > MAX_FEE:
        reasons.append(f"管理费 ({fund_fee:.2f}%) > {MAX_FEE}%")

    # 获取持仓
    holdings_data = get_fund_stock_positions(code)
    
    # 计算贝塔
    fund_beta = 0
    fund_alpha = 0
    if not index_df.empty:
        try:
            fund_beta, fund_alpha = calculate_beta(daily_returns, index_df['daily_return'])
        except Exception as e:
            print(f"    调试: 计算贝塔和阿尔法时发生异常: {e}", flush=True)

    # 计算综合评分
    score = 0
    if not reasons:
        score = (annual_return / MIN_RETURN) * 0.4 + \
                ((MAX_VOLATILITY - volatility) / MAX_VOLATILITY) * 0.3 + \
                (sharpe_ratio / MIN_SHARPE) * 0.2 + \
                (fund_beta / 1.5) * 0.1 # 假设beta理想范围为0-1.5

        if fund_beta > 0:
            score += fund_alpha / 100 # 将alpha作为加分项
            
    if not reasons:
        debug_info['筛选状态'] = '通过'
        print(f"    √ 通过筛选，评分: {score:.2f}", flush=True)
        return {
            '基金代码': code,
            '基金名称': name,
            '基金类型': fund_type,
            '年化收益率 (%)': annual_return,
            '波动率 (%)': volatility,
            '夏普比率': sharpe_ratio,
            '最新净值': latest_net_value,
            '评分': score,
            '贝塔': fund_beta,
            '阿尔法 (%)': fund_alpha,
            '管理费 (%)': fund_fee,
            '持仓股票': holdings_data['持仓股票'],
            '行业分布': holdings_data['行业分布'],
            '行业集中度 (%)': holdings_data['行业集中度 (%)']
        }
    else:
        debug_info['筛选状态'] = '未通过'
        print(f"    × 未通过筛选。原因：{' / '.join(reasons)}", flush=True)
        return None

# 主函数
def main():
    print(">>> 基金筛选工具启动...", flush=True)
    
    # 步骤 1: 获取基金列表
    funds_df = get_all_funds_from_eastmoney()
    if funds_df.empty:
        print(">>> 无法获取基金列表，程序退出。", flush=True)
        return

    # 步骤 2: 获取指数数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365) # 近一年数据
    index_code = '000300' # 沪深300指数
    index_df, _, _ = get_fund_net_values(index_code, start_date, end_date)
    if index_df.empty:
        print(f"    × 无法获取市场指数 {index_code} 数据，尝试备用指数。", flush=True)
        index_code_fallback = '000001' # 上证指数
        index_df, _, _ = get_fund_net_values(index_code_fallback, start_date, end_date)
        if index_df.empty:
            print(">>> 无法获取任何市场指数数据，贝塔计算功能将失效。", flush=True)
    
    if not index_df.empty:
        index_df['daily_return'] = index_df['value'].pct_change()
    
    total_funds = len(funds_df)
    print(f">>> 共 {total_funds} 只基金待处理（{', '.join(FUND_TYPE_FILTER)}）。", flush=True)
    
    # 步骤 3: 多线程处理基金
    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        futures = {executor.submit(process_fund, fund_row, start_date, end_date, index_df, total_funds, i + 1): i for i, fund_row in enumerate(funds_df.itertuples())}
        for future in tqdm(futures, total=total_funds, desc="处理基金", unit="只"):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"    × 处理基金时发生异常: {e}", flush=True)

    # 步骤 4: 保存结果
    if results:
        final_df = pd.DataFrame(results).sort_values('评分', ascending=False).reset_index(drop=True)
        final_df.index = final_df.index + 1
        print("\n--- 步骤4: 筛选完成，推荐基金列表 ---\n", flush=True)
        print(final_df.drop(columns=['持仓股票', '行业分布']), flush=True)
        final_df.to_csv('recommended_cn_funds.csv', index=True, index_label='排名', encoding='utf-8-sig')
        print("\n>>> 推荐结果已保存至 recommended_cn_funds.csv", flush=True)

        # 输出持仓详情
        for idx, row in final_df.iterrows():
            code = row['基金代码']
            name = row['基金名称']
            print(f"\n--- 基金 {name} ({code}) 持仓详情 ---\n", flush=True)
            if row['行业分布']:
                industry_df = pd.DataFrame(row['行业分布'])
                print(industry_df.to_string(index=False), flush=True)
                print(f"    行业集中度（前三大行业占比）: {row['行业集中度 (%)']:.2f}%", flush=True)
            else:
                print("    × 无持仓数据。", flush=True)
    else:
        print("\n>>> 未找到符合条件的基金，建议调整筛选条件。", flush=True)

if __name__ == "__main__":
    main()
