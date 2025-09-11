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
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
from datetime import datetime
import traceback

def randHeader():
    """随机生成 User-Agent，用于伪装浏览器请求"""
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

def getURL(url, tries_num=5, sleep_time=1, time_out=10, proxies=None):
    """增强型 requests，带重试机制和随机延时"""
    for i in range(tries_num):
        try:
            time.sleep(random.uniform(0.5, sleep_time))
            res = requests.get(url, headers=randHeader(), timeout=time_out, proxies=proxies)
            res.raise_for_status()
            res.encoding = 'gbk' # 强制指定编码为gbk
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功获取 {url}")
            return res
        except requests.RequestException as e:
            time.sleep(sleep_time + i * 5)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {url} 连接失败，第 {i+1} 次重试: {e}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请求 {url} 失败，已达最大重试次数")
    return None

def get_fund_name(fund_code):
    """辅助函数：从东方财富网获取基金名称"""
    try:
        fund_code = str(fund_code).zfill(6)
        url = f'http://fund.eastmoney.com/{fund_code}.html'
        res = getURL(url)
        if not res:
            return '未知基金'
        soup = BeautifulSoup(res.text, 'html.parser')
        # 修正: 基金名称可能在不同标签内
        title_elem = soup.find('div', class_='fundDetail-tit').find('h1') if soup.find('div', class_='fundDetail-tit') else None
        if not title_elem:
            title_elem = soup.find('div', class_='fund-title').find('h1') if soup.find('div', class_='fund-title') else None
        
        if title_elem:
            name = title_elem.get_text().strip().split('(')[0].strip()
            return name
        return '未知基金'
    except Exception as e:
        print(f"获取基金 {fund_code} 名称失败: {e}")
        return '未知基金'

def get_initial_fund_list():
    """从 GitHub URL 读取基金代码，并动态获取名称和排名数据"""
    url = 'https://raw.githubusercontent.com/qjlxg/rep/refs/heads/main/fund_rankings.csv'
    try:
        df = pd.read_csv(url)
        print(f"成功从 {url} 获取数据。当前列名: {df.columns.tolist()}")
        
        if 'code' in df.columns:
            df = df.rename(columns={'code': 'fund_code'})
        else:
            raise ValueError("CSV 中缺少 'code' 列")
        
        df['fund_code'] = df['fund_code'].astype(str).str.zfill(6)
        df['fund_name'] = ''
        
        funds_to_name = df['fund_code'].head(20).tolist()
        for fund_code in funds_to_name:
            name = get_fund_name(fund_code)
            df.loc[df['fund_code'] == fund_code, 'fund_name'] = name
            time.sleep(random.uniform(0.5, 1.5))
        
        df = df[['fund_code', 'fund_name', 'rank(1y)']].dropna(subset=['fund_code'])
        print(f"成功筛选出 {len(df)} 只基金，并获取了名称。")
        return df
    except Exception as e:
        print(f"从 GitHub 获取基金列表失败: {e}")
        return pd.DataFrame()

def get_fund_details(fund_code):
    """获取基金基本信息、夏普比率和最大回撤"""
    try:
        fund_code = str(fund_code).zfill(6)
        url = f'http://fund.eastmoney.com/f10/{fund_code}.html'
        
        # 修正: 使用 requests 获取网页内容，并用 pandas 读取表格
        res = getURL(url)
        if not res:
            raise Exception("请求失败，无法获取基金详情")
            
        tables = pd.read_html(res.text)
        
        # 修正: 找到正确的表格，防止因网页结构变动而失败
        details_table = None
        for table in tables:
            if '基金类型' in table.iloc[:, 0].values:
                details_table = table
                break
        if details_table is None:
            raise ValueError("未能解析到足够的基本信息表格")
        
        # 解析基本信息
        df_details = details_table.set_index(0).T
        
        # 解析风险指标
        url2 = f'http://fund.eastmoney.com/f10/tsdata_{fund_code}.html'
        res2 = getURL(url2)
        if not res2:
             raise Exception("请求失败，无法获取风险指标")
        tables2 = pd.read_html(res2.text)
        
        # 修正: 找到正确的表格，最大回撤通常是第三个表格
        if len(tables2) < 3:
            raise ValueError("未能解析到风险指标表格")
        
        df_sharpe = tables2[1].set_index(0).T
        df_drawdown = tables2[2].set_index(0).T
        
        sharpe_ratio = pd.to_numeric(df_sharpe.get('近1年', [np.nan])[0], errors='coerce')
        max_drawdown = pd.to_numeric(df_drawdown.get('近1年', [np.nan])[0], errors='coerce')
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    except Exception as e:
        print(f"获取基金 {fund_code} 详情失败: {e}")
        traceback.print_exc()
        return {
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan
        }

def get_fund_manager_info(fund_code):
    """获取基金经理的任职年限"""
    try:
        fund_code = str(fund_code).zfill(6)
        manager_url = f'http://fund.eastmoney.com/f10/jjjl_{fund_code}.html'
        res = getURL(manager_url)
        if not res:
            return np.nan
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 修正: 基金经理表格的class可能变动，使用更通用的方式定位
        manager_table = soup.find('table', class_='w780')
        if not manager_table:
            # 尝试另一种常见的表格class
            manager_table = soup.find('table', class_='tzjl')
        
        if not manager_table:
            print(f"未找到基金 {fund_code} 经理信息表格。")
            return np.nan
            
        first_row = manager_table.find_all('tr')[1]
        term_cell = first_row.find_all('td')[3]
        term_text = term_cell.get_text().strip()
        manager_term = float(re.search(r'\d+\.?\d*', term_text).group()) if re.search(r'\d+', term_text) else 0.0
        
        return manager_term
    
    except Exception as e:
        print(f"获取基金 {fund_code} 经理信息失败: {e}")
        traceback.print_exc()
        return np.nan

def get_fund_holdings_with_selenium(fund_code):
    """通过 Selenium 获取基金持仓数据"""
    fund_code = str(fund_code).zfill(6)
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument(f'user-agent={randHeader()["User-Agent"]}')
    
    # 修正: 确保路径正确
    service = Service(ChromeDriverManager().install())
    
    driver = None
    try:
        driver = webdriver.Chrome(service=service, options=options)
        url = f'http://fundf10.eastmoney.com/ccmx_{fund_code}.html'
        driver.get(url)
        
        wait = WebDriverWait(driver, 30) # 增加等待时间
        # 修正: 更新持仓表格的定位器
        table_path = '//*[@id="cctable"]'
        wait.until(EC.presence_of_element_located((By.XPATH, table_path)))
        
        soup = BeautifulSoup(driver.page_source, 'lxml')
        # 修正: 根据最新页面结构，定位表格
        holdings_table_div = soup.find('div', id='cctable')
        if not holdings_table_div:
            raise NoSuchElementException("无法找到持仓表格 div")
        
        holdings_table = holdings_table_div.find('table')
        if not holdings_table:
            raise NoSuchElementException("无法找到持仓表格 table")
            
        holdings_data = pd.read_html(str(holdings_table))[0]
        holdings_data.columns = ['排名', '股票代码', '股票名称', '最新价', '涨跌幅', '市值(万元)', '占净值比例', '持股数(万股)', '持仓市值(万元)']
        
        holdings_data['占净值比例'] = holdings_data['占净值比例'].str.strip('%').astype(float)
        
        top_10_concentration = holdings_data['占净值比例'].sum()
        num_holdings = len(holdings_data)
        
        return {'concentration': top_10_concentration, 'num_holdings': num_holdings}
    
    except (WebDriverException, TimeoutException, NoSuchElementException) as e:
        print(f"在获取基金持仓数据时发生 Selenium 错误: {e}")
        traceback.print_exc()
        return {'concentration': np.nan, 'num_holdings': np.nan}
    except Exception as e:
        print(f"在获取基金持仓数据时发生其他错误: {e}")
        traceback.print_exc()
        return {'concentration': np.nan, 'num_holdings': np.nan}
    finally:
        if driver:
            driver.quit()

def calculate_composite_score(df):
    """根据你的偏好，计算综合评分并生成最终报告"""
    print("开始进行量化评分...")
    
    # 过滤掉所有关键指标都为空的行
    df = df.dropna(subset=['sharpe_ratio', 'max_drawdown', 'manager_term', 'concentration'], how='all')
    if df.empty:
        return pd.DataFrame()

    # 归一化所有关键指标到 [0, 1] 范围，以便进行加权计算
    df['sharpe_score'] = (df['sharpe_ratio'] - df['sharpe_ratio'].min()) / (df['sharpe_ratio'].max() - df['sharpe_ratio'].min())
    df['max_drawdown_score'] = 1 - (df['max_drawdown'] - df['max_drawdown'].min()) / (df['max_drawdown'].max() - df['max_drawdown'].min())
    df['manager_term_score'] = (df['manager_term'] - df['manager_term'].min()) / (df['manager_term'].max() - df['manager_term'].min())
    df['concentration_score'] = 1 - (df['concentration'] - df['concentration'].min()) / (df['concentration'].max() - df['concentration'].min())
    
    weights = {
        'sharpe_score': 0.30,
        'max_drawdown_score': 0.20,
        'manager_term_score': 0.20,
        'concentration_score': 0.10,
    }

    if 'rank(1y)' in df.columns:
        df['ranking_score'] = 1 - (df['rank(1y)'] - df['rank(1y)'].min()) / (df['rank(1y)'].max() - df['rank(1y)'].min())
        weights['ranking_score'] = 0.20
    else:
        weights = {
            'sharpe_score': 0.35,
            'max_drawdown_score': 0.25,
            'manager_term_score': 0.25,
            'concentration_score': 0.15,
        }
        print("警告：缺少收益排名数据，已调整权重以补偿。")

    df['综合评分'] = 0
    for col, weight in weights.items():
        if col in df.columns:
            df['综合评分'] += df[col] * weight
        
    df = df.sort_values(by='综合评分', ascending=False)
    
    final_cols = [
        'fund_code', 'fund_name', '综合评分',
        'sharpe_ratio', 'max_drawdown',
        'manager_term',
        'concentration'
    ]
    
    return df[final_cols]

def main():
    """主函数，编排整个筛选流程"""
    print("第一步：开始从 GitHub 获取基金列表...")
    df = get_initial_fund_list()
    if df.empty:
        return

    all_funds_data = []

    funds_to_process = df['fund_code'].head(20).tolist()
    
    for i, fund_code in enumerate(funds_to_process, 1):
        fund_name = df[df['fund_code'] == fund_code]['fund_name'].iloc[0]
        print(f"\n[{i}/{len(funds_to_process)}] 正在深度分析基金：{fund_name} ({fund_code})...")
        
        details = get_fund_details(fund_code)
        manager_term = get_fund_manager_info(fund_code)
        holdings = get_fund_holdings_with_selenium(fund_code)
        
        fund_data = {
            'fund_code': fund_code,
            'fund_name': fund_name,
            **details,
            'manager_term': manager_term,
            **holdings
        }
        all_funds_data.append(fund_data)
        
    deep_data_df = pd.DataFrame(all_funds_data)
    
    final_df = df.merge(deep_data_df, on=['fund_code', 'fund_name'], how='left')
    
    final_report = calculate_composite_score(final_df)
    
    if final_report.empty:
        print("\n未能获取任何基金的深度数据，无法进行评分。")
        return
        
    report_path = 'advanced_fund_report.csv'
    final_report.to_csv(report_path, encoding='gbk', index=False)
    print(f"\n最终推荐基金报告已生成并保存至 '{report_path}'")
    print("请打开该文件以查看按综合评分排序的基金列表。")

if __name__ == '__main__':
    main()
