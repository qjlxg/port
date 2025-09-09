import pandas as pd
import requests
import numpy as np
import json
import re
from datetime import datetime
import time
import random
import asyncio
import aiohttp
from requests.exceptions import RequestException
from io import StringIO
from typing import List, Dict, Any, Tuple, Optional

# 筛选条件
MIN_RETURN = 3.0  # 年化收益率 ≥ 3%
MAX_VOLATILITY = 25.0  # 波动率 ≤ 25%
MIN_SHARPE = 0.2  # 夏普比率 ≥ 0.2
MAX_FEE = 2.5  # 管理费 ≤ 2.5% (仅用于信息展示)
RISK_FREE_RATE = 3.0  # 无风险利率 3%
MIN_DAYS = 100  # 最低数据天数

# 随机 User-Agent 列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'
]

async def fetch_web_data_async(session: aiohttp.ClientSession, url: str) -> Tuple[Optional[str], Optional[str]]:
    """通用异步网页数据抓取函数"""
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            response.raise_for_status()
            return await response.text(), None
    except aiohttp.ClientError as e:
        return None, f"请求失败: {e}"

async def get_fund_history_data_async(session: aiohttp.ClientSession, code: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """异步获取基金历史净值数据"""
    url = f"http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={code}&page=1&per=10000"
    html, error = await fetch_web_data_async(session, url)
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
        return df, None
    except Exception as e:
        return None, f"解析历史数据失败: {e}"

async def get_fund_realtime_info_async(session: aiohttp.ClientSession, code: str) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[str], Optional[str]]:
    """异步获取基金实时估值和名称"""
    url = f"http://fundgz.fund.eastmoney.com/Fundgz.ashx?type=js&code={code}"
    response, error = await fetch_web_data_async(session, url)
    if error:
        return None, None, None, None, error

    try:
        json_str = response.replace('jsonpgz(', '').replace(');', '')
        data = json.loads(json_str)
        
        name = data.get('name')
        gszzl = data.get('gszzl')
        
        latest_net_value = float(data.get('dwjz'))
        realtime_estimate = latest_net_value * (1 + float(gszzl) / 100) if gszzl and gszzl != '' else None
        
        return name, realtime_estimate, latest_net_value, "pingzhongdata", None
    except Exception as e:
        return None, None, None, None, f"获取实时数据失败: {e}"

async def get_fund_fee_async(session: aiohttp.ClientSession, code: str) -> Tuple[Optional[float], Optional[str]]:
    """异步获取基金管理费"""
    url = f"http://fund.eastmoney.com/{code}.html"
    html, error = await fetch_web_data_async(session, url)
    if error:
        return None, error
    
    try:
        match = re.search(r'管理费率：(\d+\.\d+)%', html)
        if match:
            fee = float(match.group(1))
            return fee, None
        else:
            return None, "无法获取管理费率"
    except Exception as e:
        return None, f"解析管理费失败: {e}"

def calculate_fund_metrics(df_history: pd.DataFrame, risk_free_rate: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """计算基金指标"""
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

def load_fund_list(file_path: str = 'fund_codes.txt') -> List[str]:
    """从文件中读取基金代码列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误：未能找到基金代码文件 {file_path}。请先运行 get_fund_list.py。", flush=True)
        return []

async def process_fund(code: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """单个基金的异步处理协程"""
    async with semaphore:
        debug_info = {'基金代码': code}
        
        print(f"正在处理基金：{code}", flush=True)
        
        name, realtime_estimate, latest_net_value, data_source, realtime_error = await get_fund_realtime_info_async(session, code)
        if realtime_error:
            debug_info['失败原因'] = f"实时数据获取失败: {realtime_error}"
            return None, debug_info

        fee, fee_error = await get_fund_fee_async(session, code)
        if fee_error:
            debug_info['失败原因'] = f"管理费获取失败: {fee_error}"
            return None, debug_info

        df_history, history_error = await get_fund_history_data_async(session, code)
        if history_error:
            debug_info['失败原因'] = f"历史数据获取失败: {history_error}"
            return None, debug_info
        
        metrics, metrics_error = calculate_fund_metrics(df_history, RISK_FREE_RATE)
        if metrics_error:
            debug_info['失败原因'] = f"指标计算失败: {metrics_error}"
            return None, debug_info
            
        debug_info.update({
            '基金名称': name,
            '最新净值': latest_net_value,
            '实时估值': round(realtime_estimate, 4) if realtime_estimate is not None else 'N/A',
            '数据来源': data_source,
            '管理费 (%)': fee,
            '数据条数': len(df_history),
            '数据开始日期': df_history['净值日期'].iloc[0],
            '数据结束日期': df_history['净值日期'].iloc[-1],
            '年化收益率 (%)': metrics['annual_return'],
            '年化波动率 (%)': metrics['volatility'],
            '夏普比率': metrics['sharpe']
        })

        if (metrics['annual_return'] >= MIN_RETURN and
            metrics['volatility'] <= MAX_VOLATILITY and
            metrics['sharpe'] >= MIN_SHARPE):
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
                     0.3 * metrics['sharpe'])
            result['综合评分'] = round(score, 2)
            debug_info['筛选结果'] = '通过'
            return result, debug_info
        else:
            debug_info['筛选结果'] = '未通过'
            return None, debug_info

async def main():
    fund_list = load_fund_list()
    if not fund_list:
        return

    print(f"已加载 {len(fund_list)} 个基金代码，开始并发处理。", flush=True)

    results = []
    debug_data = []

    # 限制并发任务数量，例如 50 个
    semaphore = asyncio.Semaphore(50)
    
    conn = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [process_fund(code, session, semaphore) for code in fund_list]
        
        # 使用 asyncio.gather 来同时运行所有任务
        processed_data = await asyncio.gather(*tasks, return_exceptions=True)
        
    for item in processed_data:
        if isinstance(item, Exception):
            print(f"任务执行中出现异常: {item}", flush=True)
            continue
            
        result, debug_info = item
        if result:
            results.append(result)
        debug_data.append(debug_info)
    
    # 保存调试信息
    debug_df = pd.DataFrame(debug_data)
    debug_df.to_csv('debug_fund_metrics.csv', index=False, encoding='utf-8-sig')
    print("\n调试信息已保存至 debug_fund_metrics.csv", flush=True)

    # 输出结果
    if results:
        final_df = pd.DataFrame(results).sort_values('综合评分', ascending=False)
        final_df.to_csv('recommended_cn_funds.csv', index=False, encoding='utf-8-sig')
        print(f"\n共筛选出 {len(final_df)} 只推荐基金，结果已保存至 recommended_cn_funds.csv", flush=True)
        print("\n推荐基金列表：", flush=True)
        print(final_df, flush=True)
    else:
        print("\n抱歉，没有找到符合筛选条件的基金。请尝试放宽筛选条件。", flush=True)

if __name__ == '__main__':
    asyncio.run(main())
