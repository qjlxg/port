import pandas as pd
import requests
import numpy as np
import json
import re
from datetime import datetime, timedelta
import time
import random
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import StringIO
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor, as_completed

# 筛选条件
MIN_RETURN = 3.0
MAX_VOLATILITY = 25.0
MIN_SHARPE = 0.2
MAX_FEE = 3.5
RISK_FREE_RATE = 3.0
MIN_DAYS = 100
BATCH_SIZE = 1000
MAX_WORKERS = 10
CACHE_DAYS = 1

# 配置 requests 重试机制
session = requests.Session()
retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# 随机 User-Agent 列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def fetch_web_data(url, proxies=None):
    headers = {'User-Agent': get_random_user_agent()}
    try:
        response = session.get(url, headers=headers, timeout=10, proxies=proxies)
        response.raise_for_status()
        return response.text, None
    except RequestException as e:
        return None, f"请求失败: {e}"

def load_cache(file_path, max_age_days=CACHE_DAYS):
    if not os.path.exists(file_path):
        return None
    if (datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))) > timedelta(days=max_age_days):
        return None
    try:
        return pd.read_csv(file_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"加载缓存 {file_path} 失败: {e}", flush=True)
        return None

def save_cache(df, file_path):
    try:
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"保存缓存 {file_path} 失败: {e}", flush=True)

def get_fund_history_data(code):
    cache_file = f"cache/history_{code}.csv"
    cached_data = load_cache(cache_file)
    if cached_data is not None:
        return cached_data, None

    url = f"http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={code}&page=1&per=10000"
    html, error = fetch_web_data(url)
    if error:
        return None, error

    match = re.search(r'<tbody>(.*?)</tbody>', html, re.DOTALL)
    if not match:
        return None, "解析历史数据失败: 无法找到表格数据"

    table_html = f"<table><thead><tr><th>净值日期</th><th>单位净值</th><th>累计净值</th><th>日增长率</th><th>申购状态</th><th>赎回状态</th><th>分红送配</th></tr></thead><tbody>{match.group(1)}</tbody></table>"

    try:
        df = pd.read_html(StringIO(table_html), header=0, encoding='utf-8')[0]
        df = df.iloc[::-1]
        df.columns = ['净值日期', '单位净值', '累计净值', '日增长率', '申购状态', '赎回状态', '分红送配']
        df['单位净值'] = pd.to_numeric(df['单位净值'], errors='coerce')
        df['累计净值'] = pd.to_numeric(df['累计净值'], errors='coerce')
        df = df.dropna(subset=['单位净值']).reset_index(drop=True)
        save_cache(df, cache_file)
        return df, None
    except Exception as e:
        return None, f"解析历史数据失败: {e}"

def get_fund_realtime_info(code):
    cache_file = f"cache/realtime_{code}.csv"
    cached_data = load_cache(cache_file)
    if cached_data is not None:
        return (cached_data['基金名称'].iloc[0], cached_data['实时估值'].iloc[0],
                cached_data['最新净值'].iloc[0], cached_data['数据来源'].iloc[0], None)

    url = f"http://fundgz.fund.eastmoney.com/Fundgz.ashx?type=js&code={code}"
    response, error = fetch_web_data(url)
    if error:
        return None, None, None, None, error

    try:
        json_str = response.replace('jsonpgz(', '').replace(');', '')
        data = json.loads(json_str)
        name = data.get('name')
        gszzl = data.get('gszzl')
        latest_net_value = float(data.get('dwjz'))
        realtime_estimate = latest_net_value * (1 + float(gszzl) / 100) if gszzl and gszzl != '' else None
        cache_df = pd.DataFrame([{
            '基金名称': name, '实时估值': realtime_estimate,
            '最新净值': latest_net_value, '数据来源': 'pingzhongdata'
        }])
        save_cache(cache_df, cache_file)
        return name, realtime_estimate, latest_net_value, 'pingzhongdata', None
    except Exception as e:
        return None, None, None, None, f"获取实时数据失败: {e}"

def get_fund_fee(code):
    cache_file = f"cache/fee_{code}.csv"
    cached_data = load_cache(cache_file)
    if cached_data is not None:
        return cached_data['管理费'].iloc[0], None

    url = f"http://fund.eastmoney.com/{code}.html"
    html, error = fetch_web_data(url)
    if error:
        return None, error

    try:
        match = re.search(r'管理费率：(\d+\.\d+)%', html)
        if match:
            fee = float(match.group(1))
            cache_df = pd.DataFrame([{'管理费': fee}])
            save_cache(cache_df, cache_file)
            return fee, None
        else:
            return None, "无法获取管理费率"
    except Exception as e:
        return None, f"解析管理费失败: {e}"

def calculate_fund_metrics(df_history, risk_free_rate):
    if df_history is None or df_history.empty:
        return None, "无净值数据"

    num_days = len(df_history)
    if num_days < MIN_DAYS:
        return None, f"数据天数不足{MIN_DAYS}天，跳过"

    daily_returns = df_history['单位净值'].pct_change().dropna()
    annual_return = np.power((1 + daily_returns).prod(), 252 / num_days) - 1
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate / 100) / volatility if volatility != 0 else 0

    metrics = {
        'annual_return': round(annual_return * 100, 2),
        'volatility': round(volatility * 100, 2),
        'sharpe': round(sharpe_ratio, 2),
        'data_days': num_days
    }
    return metrics, None

def load_fund_list(file_path='fund_codes.txt'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误：未能找到基金代码文件 {file_path}。请先运行 get_fund_list.py。", flush=True)
        return []

def is_otc_fund(code):
    return code.startswith('0')

def pre_screen_funds(fund_list):
    valid_funds = []
    cache_file = 'cache/pre_screened_funds.csv'
    cached_data = load_cache(cache_file)
    if cached_data is not None:
        return cached_data['基金代码'].tolist()

    # 并行处理管理费筛选
    def check_fund_fee(code):
        if not is_otc_fund(code):
            return None
        fee, error = get_fund_fee(code)
        time.sleep(random.uniform(2, 5))  # 增加延时
        if not error and fee is not None and fee <= MAX_FEE:
            return code
        return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_code = {executor.submit(check_fund_fee, code): code for code in fund_list}
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try:
                result = future.result()
                if result:
                    valid_funds.append(result)
                print(f"预筛选基金：{code} {'通过' if result else '未通过'}", flush=True)
            except Exception as e:
                print(f"预筛选基金 {code} 失败: {e}", flush=True)

    cache_df = pd.DataFrame({'基金代码': valid_funds})
    save_cache(cache_df, cache_file)
    return valid_funds

def process_fund(code):
    debug_info = {'基金代码': code}
    name, realtime_estimate, latest_net_value, data_source, realtime_error = get_fund_realtime_info(code)
    debug_info['基金名称'] = name
    debug_info['最新净值'] = latest_net_value
    debug_info['实时估值'] = round(realtime_estimate, 4) if realtime_estimate is not None else 'N/A'
    debug_info['数据来源'] = data_source
    if realtime_error:
        debug_info['失败原因'] = realtime_error
        return None, debug_info

    time.sleep(random.uniform(18, 15))  # 增加延时
    fee, fee_error = get_fund_fee(code)
    debug_info['管理费 (%)'] = fee
    if fee_error:
        debug_info['失败原因'] = fee_error
        return None, debug_info

    time.sleep(random.uniform(10, 18))  # 增加延时
    df_history, history_error = get_fund_history_data(code)
    debug_info['数据条数'] = len(df_history) if df_history is not None else 0
    debug_info['数据开始日期'] = df_history['净值日期'].iloc[0] if df_history is not None and not df_history.empty else 'N/A'
    debug_info['数据结束日期'] = df_history['净值日期'].iloc[-1] if df_history is not None and not df_history.empty else 'N/A'
    if history_error:
        debug_info['失败原因'] = history_error
        return None, debug_info

    metrics, metrics_error = calculate_fund_metrics(df_history, RISK_FREE_RATE)
    if metrics_error:
        debug_info['失败原因'] = metrics_error
        return None, debug_info

    debug_info.update({
        '年化收益率 (%)': metrics['annual_return'],
        '年化波动率 (%)': metrics['volatility'],
        '夏普比率': metrics['sharpe']
    })

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
            '实时估值': round(realtime_estimate, 4) if realtime_estimate is not None else 'N/A',
            '数据来源': data_source
        }
        score = (0.6 * (metrics['annual_return'] / 20) +
                 0.3 * metrics['sharpe'] +
                 0.1 * (2 - fee))
        result['综合评分'] = round(score, 2)
        return result, debug_info
    else:
        debug_info['失败原因'] = '未通过筛选'
        return None, debug_info

def main():
    if not os.path.exists('cache'):
        os.makedirs('cache')

    # 注释掉从文件中读取的代码
    # fund_list = load_fund_list()

    # 硬编码你想要测试的热门基金代码
    fund_list = [
        '001211', '005827', '008285', '001071', '003095',
        '005911', '001186', '001476', '004851', '009477',
        '009653', '008104', '008763', '001210', '001550',
        '002939', '002621', '001048', '005912', '005913',
        '000834', '005086', '005652', '003096', '005828'
    ]
    if not fund_list:
        return

    print("开始预筛选基金（检查管理费）...", flush=True)
    otc_fund_list = pre_screen_funds(fund_list)
    print(f"预筛选后剩余 {len(otc_fund_list)} 个场外基金。", flush=True)

    processed_codes_file = 'cache/processed_codes.csv'
    processed_codes = set(load_cache(processed_codes_file)['基金代码'].tolist()) if os.path.exists(processed_codes_file) else set()

    results = []
    debug_data = []
    batch_results_files = []

    for i in range(0, len(otc_fund_list), BATCH_SIZE):
        batch = otc_fund_list[i:i + BATCH_SIZE]
        batch = [code for code in batch if code not in processed_codes]
        if not batch:
            print(f"批次 {i//BATCH_SIZE + 1} 已全部处理，跳过。", flush=True)
            continue

        print(f"处理批次 {i//BATCH_SIZE + 1}，包含 {len(batch)} 个基金...", flush=True)
        batch_results = []
        batch_debug = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_code = {executor.submit(process_fund, code): code for code in batch}
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    result, debug_info = future.result()
                    batch_debug.append(debug_info)
                    if result:
                        batch_results.append(result)
                    processed_codes.add(code)
                except Exception as e:
                    batch_debug.append({'基金代码': code, '失败原因': f'处理失败: {e}'})
                print(f"完成处理基金：{code}", flush=True)
                time.sleep(random.uniform(2, 5))

        if batch_results:
            batch_df = pd.DataFrame(batch_results)
            batch_file = f'cache/recommended_cn_funds_batch_{i//BATCH_SIZE}.csv'
            save_cache(batch_df, batch_file)
            batch_results_files.append(batch_file)
        debug_df = pd.DataFrame(batch_debug)
        save_cache(debug_df, f'cache/debug_fund_metrics_batch_{i//BATCH_SIZE}.csv')
        pd.DataFrame({'基金代码': list(processed_codes)}).to_csv(processed_codes_file, index=False, encoding='utf-8-sig')

        results.extend(batch_results)
        debug_data.extend(batch_debug)

    if results:
        final_df = pd.DataFrame(results).sort_values('综合评分', ascending=False)
        final_df.to_csv('recommended_cn_funds.csv', index=False, encoding='utf-8-sig')
        print(f"共筛选出 {len(final_df)} 只推荐基金，结果已保存至 recommended_cn_funds.csv", flush=True)
        print("\n推荐基金列表：", flush=True)
        print(final_df, flush=True)
    else:
        print("抱歉，没有找到符合筛选条件的基金。请尝试放宽筛选条件。", flush=True)

    debug_df = pd.DataFrame(debug_data)
    debug_df.to_csv('debug_fund_metrics.csv', index=False, encoding='utf-8-sig')
    print("调试信息已保存至 debug_fund_metrics.csv", flush=True)

if __name__ == '__main__':
    main()
