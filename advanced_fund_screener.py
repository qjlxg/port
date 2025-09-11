import os
import json
import time
import pandas as pd
import re
import numpy as np
import requests
from bs4 import BeautifulSoup
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
from io import StringIO

def randHeader():
    """Generates a random User-Agent header."""
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
    """Robust requests function with retries and random delays."""
    for i in range(tries_num):
        try:
            time.sleep(random.uniform(0.5, sleep_time))
            res = requests.get(url, headers=randHeader(), timeout=time_out, proxies=proxies)
            res.raise_for_status()
            res.encoding = 'gbk'
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Successfully retrieved {url}")
            return res
        except requests.RequestException as e:
            time.sleep(sleep_time + i * 5)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {url} connection failed, retrying ({i+1}/{tries_num}): {e}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Request to {url} failed, maximum retries reached.")
    return None

def get_fund_name(fund_code):
    """Helper function to get the fund name from Eastmoney."""
    try:
        fund_code = str(fund_code).zfill(6)
        url = f'http://fund.eastmoney.com/{fund_code}.html'
        res = getURL(url)
        if not res:
            return 'Unknown Fund'
        soup = BeautifulSoup(res.text, 'html.parser')
        title_elem = soup.find('div', class_='fundDetail-tit') or soup.find('div', class_='fund-title')
        if title_elem:
            name = title_elem.find('h1').get_text().strip().split('(')[0].strip()
            return name
        return 'Unknown Fund'
    except Exception as e:
        print(f"Failed to get fund name for {fund_code}: {e}")
        return 'Unknown Fund'

def get_initial_fund_list():
    """Reads fund codes from a GitHub CSV and fetches their names."""
    url = 'https://raw.githubusercontent.com/qjlxg/rep/refs/heads/main/fund_rankings.csv'
    try:
        df = pd.read_csv(url)
        print(f"Successfully fetched data from {url}. Columns: {df.columns.tolist()}")
        
        if 'code' in df.columns:
            df = df.rename(columns={'code': 'fund_code'})
        else:
            raise ValueError("CSV is missing the 'code' column.")
        
        df['fund_code'] = df['fund_code'].astype(str).str.zfill(6)
        df['fund_name'] = ''
        
        funds_to_name = df['fund_code'].head(20).tolist()
        for fund_code in funds_to_name:
            name = get_fund_name(fund_code)
            df.loc[df['fund_code'] == fund_code, 'fund_name'] = name
            time.sleep(random.uniform(0.5, 1.5))
        
        df = df[['fund_code', 'fund_name', 'rank(1y)']].dropna(subset=['fund_code'])
        print(f"Successfully filtered and named {len(df)} funds.")
        return df
    except Exception as e:
        print(f"Failed to get initial fund list from GitHub: {e}")
        return pd.DataFrame()

def get_fund_details(fund_code):
    """
    **FIXED:** This function now uses a more robust approach to find
    and parse tables, making it resistant to minor website structure changes.
    It no longer relies on hardcoded table indices.
    """
    try:
        fund_code = str(fund_code).zfill(6)
        url_main = f'http://fund.eastmoney.com/f10/{fund_code}.html'
        url_risk = f'http://fund.eastmoney.com/f10/tsdata_{fund_code}.html'
        
        # Parse risk data from tsdata page
        res_risk = getURL(url_risk)
        if not res_risk:
            raise Exception("Request failed, unable to get risk data.")

        # Use StringIO to suppress the FutureWarning
        risk_tables = pd.read_html(StringIO(res_risk.text))
        
        sharpe_ratio = np.nan
        max_drawdown = np.nan
        
        # Search for tables by keyword
        for table in risk_tables:
            if '夏普比率' in table.iloc[:, 0].values:
                sharpe_ratio = pd.to_numeric(table.loc[0, '近1年'], errors='coerce')
            if '最大回撤' in table.iloc[:, 0].values:
                max_drawdown = pd.to_numeric(table.loc[0, '近1年'], errors='coerce')

        if pd.isna(sharpe_ratio) or pd.isna(max_drawdown):
            raise ValueError("Could not find required risk metrics tables.")
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    except Exception as e:
        print(f"Failed to get fund details for {fund_code}: {e}")
        traceback.print_exc()
        return {
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan
        }

def get_fund_manager_info(fund_code):
    """
    **FIXED:** This function now correctly handles the manager's term data
    by locating the correct table and extracting the text.
    """
    try:
        fund_code = str(fund_code).zfill(6)
        manager_url = f'http://fund.eastmoney.com/f10/jjjl_{fund_code}.html'
        res = getURL(manager_url)
        if not res:
            return np.nan
        
        soup = BeautifulSoup(res.text, 'html.parser')
        manager_table = soup.find('table', class_='w780') or soup.find('table', class_='tzjl')
        
        if not manager_table:
            print(f"Could not find manager info table for {fund_code}.")
            return np.nan
            
        first_row = manager_table.find_all('tr')[1]
        term_cell = first_row.find_all('td')[3]
        term_text = term_cell.get_text().strip()
        manager_term = float(re.search(r'\d+\.?\d*', term_text).group()) if re.search(r'\d+', term_text) else 0.0
        
        return manager_term
    
    except Exception as e:
        print(f"Failed to get manager info for {fund_code}: {e}")
        traceback.print_exc()
        return np.nan

def get_fund_holdings_with_selenium(fund_code):
    """
    **FIXED:** This function now correctly targets the dynamic content
    using the new `id='cctable'` selector and waits for the data to load.
    """
    fund_code = str(fund_code).zfill(6)
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument(f'user-agent={randHeader()["User-Agent"]}')
    
    service = Service(ChromeDriverManager().install())
    
    driver = None
    try:
        driver = webdriver.Chrome(service=service, options=options)
        url = f'http://fundf10.eastmoney.com/ccmx_{fund_code}.html'
        driver.get(url)
        
        wait = WebDriverWait(driver, 30)
        table_locator = (By.ID, 'cctable')
        wait.until(EC.presence_of_element_located(table_locator))
        
        soup = BeautifulSoup(driver.page_source, 'lxml')
        holdings_table_div = soup.find('div', id='cctable')
        
        if not holdings_table_div or '数据加载中' in holdings_table_div.get_text():
            raise NoSuchElementException("Holdings table data did not load.")
        
        holdings_table = holdings_table_div.find('table')
        if not holdings_table:
            raise NoSuchElementException("Unable to find the holdings table.")
            
        holdings_data = pd.read_html(StringIO(str(holdings_table)))[0]
        
        # Rename columns to avoid key errors in subsequent steps
        holdings_data.columns = ['Rank', 'StockCode', 'StockName', 'CurrentPrice', 'Change', 'MarketValue(10k)', 'NetValue%', 'Shares(10k)', 'HoldingValue(10k)']

        holdings_data['NetValue%'] = pd.to_numeric(holdings_data['NetValue%'].str.strip('%'), errors='coerce')
        
        top_10_concentration = holdings_data['NetValue%'].head(10).sum()
        num_holdings = len(holdings_data)
        
        return {'concentration': top_10_concentration, 'num_holdings': num_holdings}
    
    except (WebDriverException, TimeoutException, NoSuchElementException) as e:
        print(f"Selenium error while getting fund holdings: {e}")
        traceback.print_exc()
        return {'concentration': np.nan, 'num_holdings': np.nan}
    except Exception as e:
        print(f"An error occurred while getting fund holdings: {e}")
        traceback.print_exc()
        return {'concentration': np.nan, 'num_holdings': np.nan}
    finally:
        if driver:
            driver.quit()

def calculate_composite_score(df):
    """Calculates the composite score for funds."""
    print("Starting quantitative scoring...")
    
    df = df.dropna(subset=['sharpe_ratio', 'max_drawdown', 'manager_term', 'concentration'], how='all')
    if df.empty:
        print("No valid fund data to score.")
        return pd.DataFrame()

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
        print("Warning: Missing return ranking data. Weights adjusted to compensate.")

    df['综合评分'] = df.apply(lambda row: sum(row[col] * weight for col, weight in weights.items() if not pd.isna(row[col])), axis=1)
    
    df = df.sort_values(by='综合评分', ascending=False)
    
    final_cols = [
        'fund_code', 'fund_name', '综合评分',
        'sharpe_ratio', 'max_drawdown',
        'manager_term',
        'concentration'
    ]
    
    return df[final_cols]

def main():
    """Main function to orchestrate the entire process."""
    print("Step 1: Starting to get fund list from GitHub...")
    df = get_initial_fund_list()
    if df.empty:
        return

    all_funds_data = []

    funds_to_process = df['fund_code'].head(20).tolist()
    
    for i, fund_code in enumerate(funds_to_process, 1):
        fund_name = df[df['fund_code'] == fund_code]['fund_name'].iloc[0]
        print(f"\n[{i}/{len(funds_to_process)}] Analyzing fund: {fund_name} ({fund_code})...")
        
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
        print("\nFailed to get deep data for any funds; unable to score.")
        return
        
    report_path = 'advanced_fund_report.csv'
    final_report.to_csv(report_path, encoding='gbk', index=False)
    print(f"\nFinal report saved to '{report_path}'.")
    print("Open the file to see the ranked fund list.")

if __name__ == '__main__':
    main()
