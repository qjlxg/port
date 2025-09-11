import os
import json
import time
import pandas as pd
import re
import numpy as np
import requests
from bs4 import BeautifulSoup
from lxml import etree
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import yfinance as yf
import akshare as ak
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import random
from datetime import datetime, date, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import traceback

# 忽略警告
warnings.filterwarnings('ignore')

# --- 全局配置 ---
MIN_RETURN = 3.0  # 年化收益率 ≥ 3%
MAX_VOLATILITY = 25.0  # 波动率 ≤ 25%
MIN_SHARPE = 0.2  # 夏普比率 ≥ 0.2
MIN_DAYS = 100  # 最低数据天数
TIMEOUT = 10  # 网络请求超时时间（秒）
FUND_TYPE_FILTER = ['混合型', '股票型']  # 基金类型筛选
RISK_FREE_RATE = 0.03 # 无风险利率
START_DATE_ANALYSIS = '2023-09-11'
END_DATE_ANALYSIS = datetime.today().strftime('%Y-%m-%d')

# 配置 requests 重试机制
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

def randHeader():
    """随机生成 User-Agent"""
    head_user_agent = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36',
    ]
    return {
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'User-Agent': random.choice(head_user_agent),
        'Referer': 'http://fund.eastmoney.com/'
    }

def getURL(url, tries_num=5, sleep_time=1, time_out=TIMEOUT, proxies=None):
    """增强型 requests，带重试机制和随机延时"""
    for i in range(tries_num):
        try:
            time.sleep(random.uniform(0.5, sleep_time))
            res = session.get(url, headers=randHeader(), timeout=time_out, proxies=proxies)
            res.raise_for_status()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功获取 {url}")
            return res
        except requests.RequestException as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {url} 连接失败，第 {i+1} 次重试: {e}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请求 {url} 失败，已达最大重试次数")
    return None

def get_fund_basic_info():
    """获取全量基金基本信息"""
    print(">>> 正在获取全量基金代码和基本信息...")
    try:
        url = 'http://fund.eastmoney.com/js/fundcode_search.js'
        response = getURL(url)
        if response:
            text = re.findall(r'"(\d*?)","(.*?)","(.*?)","(.*?)","(.*?)"', response.text)
            fund_codes = [item[0] for item in text]
            fund_names = [item[2] for item in text]
            fund_types = [item[3] for item in text]
        else:
            print("天天基金基本信息获取失败，尝试 akshare...")
            fund_info = ak.fund_open_fund_info_em()
            fund_codes = fund_info['基金代码'].tolist()
            fund_names = fund_info['基金简称'].tolist()
            fund_types = fund_info['类型'].tolist()
    except Exception as e:
        print(f"获取基金基本信息失败: {e}")
        fund_info = ak.fund_open_fund_info_em()
        fund_codes = fund_info['基金代码'].tolist()
        fund_names = fund_info['基金简称'].tolist()
        fund_types = fund_info['类型'].tolist()
    
    fund_info = pd.DataFrame({
        '代码': fund_codes,
        '名称': fund_names,
        '类型': fund_types
    })
    
    excel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_fund_basic_info.xlsx')
    fund_info.to_excel(excel_path, sheet_name='基金信息', index=False)
    print(f"基金基本信息已保存至 '{excel_path}'")
    
    codes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_fund_codes.csv')
    fund_codes_df = pd.DataFrame({'基金代码': fund_codes})
    fund_codes_df.to_csv(codes_path, index=False, encoding='utf-8')
    print(f"全量基金代码列表已保存至 '{codes_path}'（{len(fund_codes)} 只基金）")
    
    return fund_info

def get_fund_data(code, sdate=START_DATE_ANALYSIS, edate=END_DATE_ANALYSIS, proxies=None):
    """获取历史净值"""
    url = f'https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={code}&page=1&per=65535&sdate={sdate}&edate={edate}'
    try:
        response = getURL(url, proxies=proxies)
        tree = etree.HTML(response.text)
        rows = tree.xpath("//tbody/tr")
        if not rows:
            raise ValueError("未找到净值表格")
        
        data = []
        for row in rows:
            cols = row.xpath("./td/text()")
            if len(cols) >= 7:
                data.append({
                    '净值日期': cols[0].strip(),
                    '单位净值': cols[1].strip(),
                    '日增长率': cols[3].strip()
                })
        
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("解析数据为空")
        
        df['净值日期'] = pd.to_datetime(df['净值日期'], format='mixed', errors='coerce')
        df['单位净值'] = pd.to_numeric(df['单位净值'], errors='coerce')
        df['日增长率'] = pd.to_numeric(df['日增长率'].str.strip('%'), errors='coerce') / 100
        df = df.dropna(subset=['净值日期', '单位净值'])
        df = df.sort_values('净值日期').reset_index(drop=True)
        
        return df
    except Exception as e:
        print(f"lxml 解析 {code} 失败 ({e})，尝试 akshare...")
        try:
            df = ak.fund_em_open_fund_daily_em(symbol=code)
            if df.empty:
                raise ValueError("akshare 数据为空")
            df = df.rename(columns={'净值日期': '净值日期', '单位净值': '单位净值', '日增长率': '日增长率'})
            df['净值日期'] = pd.to_datetime(df['净值日期'])
            df = df[(df['净值日期'] >= sdate) & (df['净值日期'] <= edate)]
            return df
        except Exception as e:
            print(f"akshare 获取 {code} 失败: {e}")
            return pd.DataFrame()

def get_fund_rankings(fund_type='hh'):
    """获取基金排名数据"""
    periods = {
        '3y': (f"{int(END_DATE_ANALYSIS[:4])-3}{END_DATE_ANALYSIS[4:]}", END_DATE_ANALYSIS),
        '2y': (f"{int(END_DATE_ANALYSIS[:4])-2}{END_DATE_ANALYSIS[4:]}", END_DATE_ANALYSIS),
        '1y': (f"{int(END_DATE_ANALYSIS[:4])-1}{END_DATE_ANALYSIS[4:]}", END_DATE_ANALYSIS),
        '6m': ((datetime.strptime(END_DATE_ANALYSIS, '%Y-%m-%d') - timedelta(days=180)).strftime('%Y-%m-%d'), END_DATE_ANALYSIS),
        '3m': ((datetime.strptime(END_DATE_ANALYSIS, '%Y-%m-%d') - timedelta(days=90)).strftime('%Y-%m-%d'), END_DATE_ANALYSIS)
    }
    all_data = []
    
    for period, (sd, ed) in periods.items():
        url = f'http://fund.eastmoney.com/data/rankhandler.aspx?op=dy&dt=kf&ft={fund_type}&rs=&gs=0&sc=qjzf&st=desc&sd={sd}&ed={ed}&es=1&qdii=&pi=1&pn=10000&dx=1'
        try:
            response = getURL(url)
            if not response: continue
            
            content = response.text.strip()
            match = re.search(r'datas:(.*?),allRecords:(\d+),', content)
            if not match: continue
            
            datas_str = match.group(1)
            all_records = int(match.group(2))
            
            records_str = datas_str.replace('[', '').replace(']', '')
            records = [item.strip() for item in records_str.split('","')]
            
            if not records: continue
            
            df = pd.DataFrame([r.split(',') for r in records])
            df = df[[0, 1, 3]].rename(columns={0: '代码', 1: '名称', 3: f'rose({period})'})
            df[f'rose({period})'] = pd.to_numeric(df[f'rose({period})'].str.replace('%', ''), errors='coerce') / 100
            df[f'rank({period})'] = range(1, len(df) + 1)
            df[f'rank_r({period})'] = df[f'rank({period})'] / all_records
            df.set_index('代码', inplace=True)
            all_data.append(df)
            print(f"获取 {period} 排名数据：{len(df)} 条（总计 {all_records}）")
        except Exception as e:
            print(f"获取 {period} 排名失败: {e}")
    
    if all_data:
        df_final = all_data[0]
        for df in all_data[1:]:
            df_final = df_final.join(df, how='inner')
        return df_final
    return pd.DataFrame()

def apply_4433_rule(df, total_records):
    """应用四四三三法则筛选"""
    thresholds = {
        '3y': 0.25, '2y': 0.25, '1y': 0.25,
        '6m': 1/3, '3m': 1/3
    }
    filtered_df = df.copy()
    for period in thresholds:
        rank_col = f'rank_r({period})'
        if rank_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[rank_col] <= thresholds[period]]
    print(f"四四三三法则筛选出 {len(filtered_df)} 只基金")
    return filtered_df

def get_detailed_info(code):
    """获取基金详细信息，包括持仓和基金经理"""
    try:
        url = f'http://fundf10.eastmoney.com/f10/ccmx_{code}.html'
        res = getURL(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 基金经理
        manager_url = f'http://fund.eastmoney.com/f10/jjjl_{code}.html'
        manager_res = getURL(manager_url)
        manager_soup = BeautifulSoup(manager_res.text, 'html.parser')
        manager_info = manager_soup.find('div', class_='jl_box')
        manager_name = manager_info.find('a').text if manager_info and manager_info.find('a') else 'N/A'
        
        # 基金持仓
        holdings = []
        tables = soup.find_all('table')
        if tables:
            for row in tables[0].find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) >= 5:
                    holdings.append({
                        '股票名称': cols[1].text.strip(),
                        '占净值比例(%)': cols[2].text.strip(),
                    })
        
        return {
            '代码': code,
            '基金经理': manager_name,
            '前十大持仓': json.dumps(holdings, ensure_ascii=False)
        }
    except Exception as e:
        print(f"获取基金 {code} 详细信息失败: {e}")
        return {
            '代码': code,
            '基金经理': 'N/A',
            '前十大持仓': 'N/A'
        }

def analyze_fund(fund_code):
    """分析基金风险参数并生成图表"""
    print(f"正在分析基金 {fund_code}...")
    data = get_fund_data(fund_code, sdate=START_DATE_ANALYSIS, edate=END_DATE_ANALYSIS)
    
    if data.empty or len(data) < MIN_DAYS:
        return {"error": "无法获取足够数据"}
    
    returns = data['日增长率'].dropna()
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    mdd = drawdown.min()
    
    sharpe_ratio = (annual_return - RISK_FREE_RATE) / annual_volatility if annual_volatility > 0 else 0
    
    start_value = data['单位净值'].iloc[0]
    end_value = data['单位净值'].iloc[-1]
    years = (data['净值日期'].iloc[-1] - data['净值日期'].iloc[0]).days / 365.25
    cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else 0
    
    # 可视化
    fig = go.Figure(data=[
        go.Scatter(x=data['净值日期'], y=data['单位净值'], name='单位净值', line=dict(color='royalblue', width=2)),
    ])
    fig.update_layout(title=f'基金 {fund_code} 净值走势', xaxis_title='日期', yaxis_title='净值')
    plot_path = f'fund_plot_{fund_code}.html'
    fig.write_html(plot_path)
    print(f"净值走势图已保存至 '{plot_path}'")
    
    plt.figure(figsize=(10, 5))
    sns.histplot(returns, bins=30, kde=True)
    plt.title(f'基金 {fund_code} 收益率分布')
    plt.xlabel('收益率')
    plt.ylabel('频率')
    hist_path = f'fund_return_hist_{fund_code}.png'
    plt.savefig(hist_path)
    plt.close()
    print(f"收益率分布图已保存至 '{hist_path}'")

    return {
        "fund_code": fund_code,
        "年化收益率 (%)": f"{annual_return * 100:.2f}",
        "年化波动率 (%)": f"{annual_volatility * 100:.2f}",
        "最大回撤 (%)": f"{mdd * 100:.2f}",
        "夏普比率": f"{sharpe_ratio:.2f}",
        "CAGR (%)": f"{cagr * 100:.2f}",
        "推荐": "值得考虑购买" if sharpe_ratio > 1.0 and mdd > -0.20 else "可观察"
    }

def comprehensive_screener():
    """综合筛选器主函数"""
    print("--- 启动基金综合筛选器 ---")
    
    # 1. 获取全量基金基本信息
    fund_info_df = get_fund_basic_info()
    filtered_by_type = fund_info_df[fund_info_df['类型'].isin(FUND_TYPE_FILTER)].reset_index(drop=True)
    print(f"根据类型筛选后，剩余 {len(filtered_by_type)} 只基金")

    # 2. 获取基金排名数据
    rank_df = get_fund_rankings(fund_type='hh')
    if rank_df.empty:
        print("排名数据获取失败，跳过四四三三法则筛选。")
        screened_by_4433 = filtered_by_type
    else:
        rank_df.reset_index(inplace=True)
        merged_df = pd.merge(filtered_by_type, rank_df, left_on='代码', right_on='代码', how='inner')
        total_records = len(merged_df)
        screened_by_4433 = apply_4433_rule(merged_df, total_records)
    
    # 3. 进一步筛选并计算风险指标
    results = []
    fund_codes_to_analyze = screened_by_4433['代码'].unique().tolist()
    
    # 使用多线程加速
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_fund, code): code for code in fund_codes_to_analyze}
        for future in futures:
            try:
                result = future.result()
                if not result.get("error"):
                    results.append(result)
            except Exception as e:
                print(f"    × 处理基金时发生异常: {e}")
                traceback.print_exc()

    if not results:
        print("\n>>> 未找到符合条件的基金，建议调整筛选条件。")
        return pd.DataFrame()

    final_df = pd.DataFrame(results)
    
    # 4. 应用量化指标筛选
    final_df['年化收益率 (%)'] = pd.to_numeric(final_df['年化收益率 (%)'], errors='coerce')
    final_df['年化波动率 (%)'] = pd.to_numeric(final_df['年化波动率 (%)'], errors='coerce')
    final_df['夏普比率'] = pd.to_numeric(final_df['夏普比率'], errors='coerce')
    final_df = final_df[
        (final_df['年化收益率 (%)'] >= MIN_RETURN) &
        (final_df['年化波动率 (%)'] <= MAX_VOLATILITY) &
        (final_df['夏普比率'] >= MIN_SHARPE)
    ]
    final_df = final_df.sort_values('年化收益率 (%)', ascending=False).reset_index(drop=True)
    
    final_df = pd.merge(final_df, screened_by_4433, left_on='fund_code', right_on='代码', how='inner')
    
    output_path = 'selected_funds.csv'
    final_df.to_csv(output_path, encoding='gbk', index=False)
    print(f"\n筛选结果已保存至 '{output_path}'（{len(final_df)} 只基金）")
    
    return final_df

def main():
    """主执行函数"""
    
    # Step 1: 运行综合筛选器，获取精选基金列表
    selected_funds_df = comprehensive_screener()
    
    if selected_funds_df.empty:
        print("没有找到符合条件的基金，程序结束。")
        return
    
    # Step 2: 对精选基金进行详细分析
    print("\n--- 开始对精选基金进行详细分析和持仓抓取 ---")
    detailed_results = []
    
    # 仅处理前 20 只基金，以节省时间
    funds_to_detail = selected_funds_df.head(20)['代码'].tolist()
    
    for code in funds_to_detail:
        print(f"正在获取基金 {code} 的详细信息...")
        detail = get_detailed_info(code)
        
        # 风险分析和图表生成
        analysis_result = analyze_fund(code)
        
        if not analysis_result.get('error'):
            # 将分析结果合并到详细信息中
            detail.update(analysis_result)
            detailed_results.append(detail)
            
            # 保存分析结果
            analysis_path = f'fund_analysis_{code}.json'
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=4, ensure_ascii=False)
            print(f"分析结果已保存至 '{analysis_path}'。")
    
    if detailed_results:
        final_detailed_df = pd.DataFrame(detailed_results)
        final_detailed_df.to_csv('final_detailed_screener_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n所有详细分析结果已保存至 'final_detailed_screener_results.csv'。")
        
        # 打印最终结果（简化版）
        print("\n--- 最终精选基金概览 ---")
        display_df = final_detailed_df[['代码', '名称', '基金经理', '夏普比率', '最大回撤 (%)', '前十大持仓']]
        print(display_df.to_string())

if __name__ == "__main__":
    main()
