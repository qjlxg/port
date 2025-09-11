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

def get_fund_rankings(fund_type='hh', start_date='2018-09-11', end_date='2025-09-11', proxies=None):
    """Fetches fund rankings and returns a filtered DataFrame."""
    all_data = []
    
    # Define ranking periods for the 4433 rule
    periods = {
        '3y': (start_date, end_date),
        '2y': (f"{int(end_date[:4])-2}{end_date[4:]}", end_date),
        '1y': (f"{int(end_date[:4])-1}{end_date[4:]}", end_date),
        '6m': (f"{int(end_date[:4])-(1 if int(end_date[5:7])<=6 else 0)}-{int(end_date[5:7])-6:02d}{end_date[7:]}", end_date),
        '3m': (f"{int(end_date[:4])-(1 if int(end_date[5:7])<=3 else 0)}-{int(end_date[5:7])-3:02d}{end_date[7:]}", end_date)
    }
    
    for period, (sd, ed) in periods.items():
        url = f'http://fund.eastmoney.com/data/rankhandler.aspx?op=dy&dt=kf&ft={fund_type}&rs=&gs=0&sc=qjzf&st=desc&sd={sd}&ed={ed}&es=1&qdii=&pi=1&pn=10000&dx=1'
        try:
            response = getURL(url, proxies=proxies)
            if not response:
                raise ValueError("Could not get response.")
            content = response.text
            content = re.sub(r'var rankData\s*=\s*({.*?});?', r'\1', content)
            content = re.sub(r'([,{])(\w+):', r'\1"\2":', content)
            content = content.replace('\'', '"')
            data = json.loads(content)
            records = data['datas']
            total = int(data['allRecords'])
            
            df = pd.DataFrame([r.split(',') for r in records])
            df = df[[0, 1, 3]].rename(columns={0: 'code', 1: 'name', 3: f'rose({period})'})
            df[f'rose({period})'] = pd.to_numeric(df[f'rose({period})'].str.replace('%', ''), errors='coerce') / 100
            df[f'rank({period})'] = range(1, len(df) + 1)
            df[f'rank_r({period})'] = df[f'rank({period})'] / total
            df.set_index('code', inplace=True)
            all_data.append(df)
            print(f"Successfully fetched {period} rankings: {len(df)} records (total {total})")
        except Exception as e:
            print(f"Failed to get {period} rankings: {e}")
            all_data.append(pd.DataFrame())

    if not all_data or all(df.empty for df in all_data):
        print("All ranking data fetching failed.")
        return pd.DataFrame()

    # Merge all ranking data
    merged_df = all_data[0].copy()
    for df in all_data[1:]:
        if not df.empty:
            merged_df = merged_df.join(df, how='outer')
    
    # Apply the 4433 rule
    rule_thresholds = {'3y': 0.25, '2y': 0.25, '1y': 0.25, '6m': 1/3, '3m': 1/3}
    filtered_df = merged_df.copy()
    for period, threshold in rule_thresholds.items():
        rank_col = f'rank_r({period})'
        if rank_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[rank_col] <= threshold]
            
    print(f"4433 rule filtered down to {len(filtered_df)} funds.")
    
    return filtered_df.reset_index().rename(columns={'index': 'fund_code', 'name': 'fund_name'})

def get_fund_details(fund_code):
    """
    **FIXED**: This function now uses a robust method to find and parse
    the risk metrics tables, no longer relying on fixed table indices.
    """
    try:
        fund_code = str(fund_code).zfill(6)
        url_risk = f'http://fund.eastmoney.com/f10/tsdata_{fund_code}.html'
        
        res_risk = getURL(url_risk)
        if not res_risk:
            raise Exception("Request failed, unable to get risk data.")

        # Use StringIO to handle text as a file for pandas
        risk_tables = pd.read_html(StringIO(res_risk.text))
        
        sharpe_ratio = np.nan
        max_drawdown = np.nan
        
        # Search for tables by keyword
        for table in risk_tables:
            if '夏普比率' in table.iloc[:, 0].values:
                sharpe_ratio = pd.to_numeric(table.loc[table.iloc[:, 0] == '夏普比率', '近1年'].iloc[0], errors='coerce')
            if '最大回撤' in table.iloc[:, 0].values:
                max_drawdown = pd.to_numeric(table.loc[table.iloc[:, 0] == '最大回撤', '近1年'].iloc[0], errors='coerce')

        if pd.isna(sharpe_ratio) or pd.isna(max_drawdown):
            raise ValueError("Could not find required risk metrics in the tables.")
        
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
    **FIXED**: This function now correctly handles the manager's term data
    by locating the correct table and extracting the text.
    """
    try:
        fund_code = str(fund_code).zfill(6)
        manager_url = f'http://fund.eastmoney.com/f10/jjjl_{fund_code}.html'
        res = getURL(manager_url)
        if not res:
            return np.nan
        
        soup = BeautifulSoup(res.text, 'html.parser')
        manager_table = soup.find('table', class_='tzjl') or soup.find('table', class_='w780')
        
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
    **FIXED**: This function correctly targets the dynamic content
    using the new id='cctable' selector and waits for the data to load.
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
        'concentration',
        'num_holdings'
    ]
    
    return df[final_cols]

def main():
    """Main function to orchestrate the entire process."""
    print("Step 1: Starting to get fund rankings and apply 4433 rule...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    
    # Get rankings and filter, the logic is now self-contained in get_fund_rankings
    filtered_df = get_fund_rankings(fund_type='hh', start_date=start_date, end_date=end_date)
    
    if filtered_df.empty:
        print("\nFailed to get rankings or no funds passed the 4433 filter. Exiting.")
        return

    # Process only the top 50 ranked funds for efficiency
    funds_to_process = filtered_df.head(50).to_dict('records')
    all_funds_data = []

    for i, fund in enumerate(funds_to_process, 1):
        fund_code = fund['fund_code']
        fund_name = fund['fund_name']
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
    
    final_df = filtered_df.merge(deep_data_df, on=['fund_code', 'fund_name'], how='left')
    
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
