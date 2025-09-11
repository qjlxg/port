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
import pymysql
import plotly.graph_objects as go
from pymongo import MongoClient
import random
from datetime import datetime

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

def getURL(url, tries_num=5, sleep_time=1, time_out=10, proxies=None):
    """增强型 requests，带重试机制和随机延时"""
    for i in range(tries_num):
        try:
            time.sleep(random.uniform(0.5, sleep_time))  # 随机延时
            res = requests.get(url, headers=randHeader(), timeout=time_out, proxies=proxies)
            res.raise_for_status()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功获取 {url}")
            return res
        except requests.RequestException as e:
            time.sleep(sleep_time + i * 5)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {url} 连接失败，第 {i+1} 次重试: {e}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请求 {url} 失败，已达最大重试次数")
    return None

class PyMySQL:
    """MySQL 操作类"""
    def __init__(self, host, user, passwd, db, port=3306, charset='utf8'):
        try:
            self.db = pymysql.connect(host=host, user=user, passwd=passwd, db=db, port=port, charset=charset)
            self.db.ping(True)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] MySQL 连接成功: {user}@{host}:{port}/{db}")
            self.cur = self.db.cursor()
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] MySQL 连接失败: {e}")
            raise

    def insertData(self, table, my_dict):
        try:
            cols = ', '.join(my_dict.keys())
            values = '","'.join([str(v).replace('"', '') for v in my_dict.values()])
            sql = f"replace into {table} ({cols}) values (\"{values}\")"
            result = self.cur.execute(sql)
            self.db.commit()
            return result
        except Exception as e:
            self.db.rollback()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 数据插入失败: {e}")
            return 0

def get_fund_codes_from_csv():
    """从 CSV 读取基金代码"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fund_codes.csv')
    if os.path.exists(file_path):
        fund_code = pd.read_csv(file_path, encoding='gbk')
        return fund_code['trade_code'].tolist()
    print(f"未找到 {file_path}，将爬取全量基金代码")
    return None

def get_realtime_valuation(fund_code):
    """获取实时估值（第一篇文章）"""
    try:
        url = f"http://fundgz.1234567.com.cn/js/{fund_code}.js"
        response = getURL(url)
        if response:
            data = json.loads(response.text.replace('jsonpgz(', '').replace(');', ''))
            realtime_price = float(data['dwjz'])
            management_fee = 0.001  # 假设年化管理费 0.1%
            daily_fee = (1 + management_fee) ** (1/365) - 1
            net_value = realtime_price * (1 - daily_fee)
            return {
                'fund_code': fund_code,
                'realtime_price': realtime_price,
                'net_value': net_value,
                'valuation_date': data.get('gztime', time.strftime('%Y-%m-%d %H:%M:%S'))
            }
    except Exception as e:
        print(f"天天基金实时估值失败 ({fund_code}): {e}")
        # 回退到 yfinance（适用于国际 ETF）
        try:
            fund_data = yf.Ticker(fund_code)
            realtime_price = fund_data.history(period="1d")['Close'].iloc[-1]
            management_fee = 0.001
            daily_fee = (1 + management_fee) ** (1/365) - 1
            net_value = realtime_price * (1 - daily_fee)
            return {
                'fund_code': fund_code,
                'realtime_price': realtime_price,
                'net_value': net_value,
                'valuation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            print(f"yfinance 实时估值失败 ({fund_code}): {e}")
            return None

def get_fund_rankings(fund_type='hh', start_date='2018-09-11', end_date='2025-09-11', proxies=None):
    """获取基金排名数据，优化分页处理（第二、第四篇文章）"""
    periods = {
        '3y': (start_date, end_date),
        '2y': (f"{int(end_date[:4])-2}{end_date[4:]}", end_date),
        '1y': (f"{int(end_date[:4])-1}{end_date[4:]}", end_date),
        '6m': (f"{int(end_date[:4])-(1 if int(end_date[5:7])<=6 else 0)}-{int(end_date[5:7])-6:02d}{end_date[7:]}", end_date),
        '3m': (f"{int(end_date[:4])-(1 if int(end_date[5:7])<=3 else 0)}-{int(end_date[5:7])-3:02d}{end_date[7:]}", end_date)
    }
    all_data = []
    
    for period, (sd, ed) in periods.items():
        url = f'http://fund.eastmoney.com/data/rankhandler.aspx?op=dy&dt=kf&ft={fund_type}&rs=&gs=0&sc=qjzf&st=desc&sd={sd}&ed={ed}&es=1&qdii=&pi=1&pn=10000&dx=1'
        try:
            response = getURL(url, proxies=proxies)
            if not response:
                raise ValueError("无法获取响应")
            content = response.text
            content = content[15:-1]
            content = content.replace('datas', '"datas"').replace('allRecords', '"allRecords"')
            data = json.loads(f"{{{content}}}")
            records = data['datas']
            total = int(data['allRecords'])
            
            df = pd.DataFrame([r.split(',') for r in records])
            df = df[[0, 1, 3]].rename(columns={0: 'code', 1: 'name', 3: f'rose({period})'})
            df[f'rose({period})'] = pd.to_numeric(df[f'rose({period})'].str.replace('%', ''), errors='coerce') / 100
            df[f'rank({period})'] = range(1, len(df) + 1)
            df[f'rank_r({period})'] = df[f'rank({period})'] / total
            df.set_index('code', inplace=True)
            all_data.append(df)
            print(f"获取 {period} 排名数据：{len(df)} 条（总计 {total}）")
        except Exception as e:
            print(f"获取 {period} 排名失败: {e}")
    
    if all_data:
        df_final = all_data[0]
        for df in all_data[1:]:
            df_final = df_final.join(df, how='inner')
        df_final.to_csv('fund_rankings.csv', encoding='gbk')
        print(f"排名数据已保存至 'fund_rankings.csv'")
        return df_final
    print("所有排名数据获取失败，将使用推荐基金列表")
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

def get_fund_details(code, proxies=None, mysql=None):
    """获取基金基本信息和夏普比率，优化数据清洗（第四、第五篇文章）"""
    try:
        url = f'http://fund.eastmoney.com/f10/{code}.html'
        tables = pd.read_html(url, encoding='utf-8')
        df = tables[1]
        df1 = df[[0, 1]].set_index(0).T
        df2 = df[[2, 3]].set_index(2).T
        df1['code'] = code
        df2['code'] = code
        df1.set_index('code', inplace=True)
        df2.set_index('code', inplace=True)
        df_details = pd.concat([df1, df2], axis=1)
        
        url2 = f'http://fund.eastmoney.com/f10/tsdata_{code}.html'
        tables2 = pd.read_html(url2, encoding='utf-8')
        df_sharpe = tables2[1]
        df_sharpe['code'] = code
        df_sharpe.set_index('code', inplace=True)
        df_sharpe.drop('基金风险指标', axis='columns', inplace=True, errors='ignore')
        df_sharpe = df_sharpe[1:]
        df_sharpe.columns = [f'夏普比率(近{c})' for c in df_sharpe.columns]
        df_sharpe = df_sharpe.apply(pd.to_numeric, errors='coerce')
        
        df_final = df_details.combine_first(df_sharpe)
        if mysql:
            result = {
                'fund_code': code,
                'fund_name': df_final.get('基金全称', ['N/A'])[0],
                'fund_type': df_final.get('基金类型', ['N/A'])[0],
                'asset_value': df_final.get('资产规模', ['N/A'])[0],
                'sharpe_3y': df_final.get('夏普比率(近3年)', ['N/A'])[0],
                'data_source': 'eastmoney',
                'created_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'updated_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'created_by': 'eastmoney',
                'updated_by': 'eastmoney'
            }
            mysql.insertData('fund_info', result)
        return df_final
    except Exception as e:
        print(f"获取基金 {code} 详情失败: {e}")
        return pd.DataFrame()

def filter_recommended_funds(csv_path='recommended_cn_funds.csv', conditions=None):
    """基于推荐基金 CSV 进行二次筛选，优化条件筛选（第五篇文章）"""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        filtered_df = df.copy()
        
        if conditions:
            for key, value in conditions.items():
                if isinstance(value, tuple):
                    filtered_df = filtered_df[(filtered_df[key] >= value[0]) & (filtered_df[key] <= value[1])]
                else:
                    filtered_df = filtered_df[filtered_df[key] >= value]
        
        filtered_df = filtered_df.sort_values(by='综合评分', ascending=False)
        output_path = 'filtered_funds.csv'
        filtered_df.to_csv(output_path, encoding='gbk', index=False)
        print(f"二次筛选结果已保存至 '{output_path}'（{len(filtered_df)} 只基金）")
        return filtered_df
    except Exception as e:
        print(f"二次筛选失败: {e}")
        return pd.DataFrame()

def get_fund_basic_info(mysql=None):
    """获取全量基金基本信息，添加 akshare 备选（第三、第六篇文章）"""
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
    
    excel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fund_basic_info.xlsx')
    try:
        fund_info.to_excel(excel_path, sheet_name='基金信息', index=False)
        print(f"基金基本信息已保存至 '{excel_path}'")
    except ImportError as e:
        print(f"保存 Excel 失败（可能缺少 openpyxl）：{e}")
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fund_basic_info.csv')
        fund_info.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"已回退保存至 '{csv_path}'")
    
    codes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_fund_codes.csv')
    fund_codes_df = pd.DataFrame({'基金代码': fund_codes})
    fund_codes_df.to_csv(codes_path, index=False, encoding='utf-8')
    print(f"全量基金代码列表已保存至 '{codes_path}'（{len(fund_codes)} 只基金）")
    
    if mysql:
        for _, row in fund_info.iterrows():
            result = {
                'fund_code': row['代码'],
                'fund_name': row['名称'],
                'fund_type': row['类型'],
                'data_source': 'eastmoney',
                'created_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'updated_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'created_by': 'eastmoney',
                'updated_by': 'eastmoney'
            }
            mysql.insertData('fund_info', result)
        print("基金基本信息已保存到 MySQL fund_info 表")
    
    return fund_info

def get_fund_data(code, sdate='', edate='', proxies=None, mysql=None, mongodb=None):
    """获取历史净值，优化分页和备选 akshare（第二、第三、第四篇文章）"""
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
                    '累计净值': cols[2].strip(),
                    '日增长率': cols[3].strip(),
                    '申购状态': cols[4].strip(),
                    '赎回状态': cols[5].strip(),
                    '分红送配': cols[6].strip()
                })
        
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("解析数据为空")
        
        # 数据清洗
        df['净值日期'] = pd.to_datetime(df['净值日期'], format='mixed', errors='coerce')
        df['单位净值'] = pd.to_numeric(df['单位净值'], errors='coerce')
        df['累计净值'] = pd.to_numeric(df['累计净值'], errors='coerce')
        df['日增长率'] = pd.to_numeric(df['日增长率'].str.strip('%'), errors='coerce') / 100
        df = df.dropna(subset=['净值日期', '单位净值'])
        
        if mysql:
            for _, row in df.iterrows():
                result = {
                    'the_date': row['净值日期'],
                    'fund_code': code,
                    'nav': row['单位净值'],
                    'add_nav': row['累计净值'],
                    'nav_chg_rate': row['日增长率'],
                    'buy_state': row['申购状态'],
                    'sell_state': row['赎回状态'],
                    'div': row['分红送配'],
                    'created_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'updated_date': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                mysql.insertData('fund_nav', result)
            print(f"基金 {code} 的净值数据已保存到 MySQL fund_nav 表")
        
        if mongodb:
            client = MongoClient('mongodb://localhost:27017/')
            db = client['fund_db']
            collection = db[f'nav_{code}']
            collection.insert_many(df.to_dict('records'))
            print(f"基金 {code} 的净值数据已保存到 MongoDB nav_{code} 集合")
        
        print(f"成功获取 {code} 的 {len(df)} 条净值数据")
        return df
    except Exception as e:
        print(f"lxml 解析失败 ({e})，尝试 akshare...")
        try:
            df = ak.fund_open_fund_daily_em(code=code)
            if df.empty:
                raise ValueError("akshare 数据为空")
            df['净值日期'] = pd.to_datetime(df['净值日期'], format='mixed', errors='coerce')
            df = df[(df['净值日期'] >= sdate) & (df['净值日期'] <= edate)]
            if mysql:
                for _, row in df.iterrows():
                    result = {
                        'the_date': row['净值日期'],
                        'fund_code': code,
                        'nav': row['单位净值'],
                        'add_nav': row.get('累计净值', None),
                        'nav_chg_rate': row.get('日增长率', None),
                        'buy_state': row.get('申购状态', 'N/A'),
                        'sell_state': row.get('赎回状态', 'N/A'),
                        'div': row.get('分红送配', 'N/A'),
                        'created_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'updated_date': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    mysql.insertData('fund_nav', result)
            if mongodb:
                client = MongoClient('mongodb://localhost:27017/')
                db = client['fund_db']
                collection = db[f'nav_{code}']
                collection.insert_many(df.to_dict('records'))
            print(f"akshare 获取 {code} 的 {len(df)} 条净值数据")
            return df
        except Exception as e:
            print(f"akshare 解析失败: {e}")
            return pd.DataFrame()

def get_fund_managers(fund_code, mysql=None):
    """获取基金经理数据，优化筛选逻辑（第五篇文章）"""
    fund_url = f'http://fund.eastmoney.com/f10/jjjl_{fund_code}.html'
    try:
        res = getURL(fund_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        tables = soup.find_all("table")
        tab = tables[1]
        result = []
        for tr in tab.find_all('tr'):
            if tr.find_all('td'):
                try:
                    manager_data = {
                        'fund_code': fund_code,
                        'start_date': tr.select('td:nth-of-type(1)')[0].get_text().strip(),
                        'end_date': tr.select('td:nth-of-type(2)')[0].get_text().strip(),
                        'fund_managers': tr.select('td:nth-of-type(3)')[0].get_text().strip(),
                        'term': tr.select('td:nth-of-type(4)')[0].get_text().strip(),
                        'management_return': tr.select('td:nth-of-type(5)')[0].get_text().strip(),
                        'management_rank': tr.select('td:nth-of-type(6)')[0].get_text().strip()
                    }
                    if mysql:
                        result_mysql = {
                            'fund_code': fund_code,
                            'manager_name': manager_data['fund_managers'],
                            'start_date': manager_data['start_date'],
                            'end_date': manager_data['end_date'],
                            'term_days': manager_data['term'].split('天')[0].strip(),
                            'data_source': 'eastmoney',
                            'created_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'updated_date': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        mysql.insertData('fund_manager', result_mysql)
                    result.append(manager_data)
                except IndexError:
                    continue
        output_filename = f"fund_managers_{fund_code}.json"
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"基金经理数据已保存至 '{output_path}'")
        return result
    except Exception as e:
        print(f"获取基金经理数据失败: {e}")
        return []

def get_fund_holdings_with_selenium(fund_code):
    """
    通过 Selenium 获取基金持仓数据。
    """
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
        time.sleep(5)  # 等待页面加载，可根据需要调整

        soup = BeautifulSoup(driver.page_source, 'lxml')
        tables = soup.find_all('table')

        if not tables:
            print(f"× 未在页面找到表格。URL: {url}")
            return []

        # 查找包含持仓数据的表格
        holdings_table = None
        for table in tables:
            if table.find('th', text='股票名称'):
                holdings_table = table
                break
        
        if not holdings_table:
            print(f"× 未在页面找到股票持仓表格。URL: {url}")
            return []

        holdings_data = []
        rows = holdings_table.find_all('tr')[1:]  # 跳过表头
        if not rows:
            print(f"× 股票持仓表格为空。URL: {url}")
            return []

        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 8:
                try:
                    # 修正索引以匹配新的网页结构
                    stock_code = cols[1].text.strip()
                    stock_name = cols[2].text.strip()
                    proportion = cols[6].text.strip().replace('%', '')
                    shares = cols[7].text.strip()
                    value = cols[8].text.strip()

                    # 数据清洗
                    if proportion == '-' or not proportion:
                        proportion = '0'
                    if shares == '-' or not shares:
                        shares = '0'
                    if value == '-' or not value:
                        value = '0'
                    
                    holdings_data.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'proportion': float(proportion),
                        'shares': float(re.sub(r'[^\d.]', '', shares)),
                        'value': float(re.sub(r'[^\d.]', '', value))
                    })
                except (IndexError, ValueError) as e:
                    print(f"解析持仓数据时发生错误：{e}")
                    continue
        
        return holdings_data
    
    except Exception as e:
        print(f"在获取基金持仓数据时发生错误: {e}")
        return []
    finally:
        if driver:
            driver.quit()

def analyze_fund(fund_code, start_date, end_date, use_yfinance=False, mongodb=None):
    """分析基金风险指标（第四篇文章）"""
    data_source = 'yfinance' if use_yfinance else 'akshare'
    if use_yfinance:
        try:
            data = yf.download(fund_code, start=start_date, end=end_date)['Close']
            returns = data.pct_change().dropna()
        except Exception as e:
            print(f"yfinance 下载 {fund_code} 数据失败: {e}")
            return {"error": "无法获取数据"}
    else:
        try:
            df = get_fund_data(fund_code, sdate=start_date, edate=end_date, mongodb=mongodb)
            returns = df['单位净值'].pct_change().dropna()
        except Exception as e:
            print(f"akshare 获取 {fund_code} 数据失败: {e}")
            return {"error": "无法获取数据"}

    if returns.empty:
        return {"error": "没有足够的回报数据进行分析"}

    try:
        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_returns - 0.03) / annual_volatility
        max_drawdown = (returns.max() - returns.min()) / returns.max()
        
        # 将 NumPy float64 转换为 Python float
        result = {
            "fund_code": fund_code,
            "annual_returns": float(annual_returns),
            "annual_volatility": float(annual_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "data_source": data_source
        }
        return result
    except Exception as e:
        print(f"分析基金 {fund_code} 风险参数失败: {e}")
        return {"error": "风险参数计算失败"}

def plot_returns(fund_code, returns):
    """绘制基金收益图表"""
    if returns.empty:
        print("没有足够的数据进行图表绘制")
        return None
    
    returns_cum = (1 + returns).cumprod()
    
    plt.figure(figsize=(10, 6))
    plt.plot(returns_cum.index, returns_cum.values, label=fund_code)
    plt.title(f'{fund_code} 累计收益率')
    plt.xlabel('日期')
    plt.ylabel('累计收益')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    chart_path = f"fund_return_hist_{fund_code}.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"收益图表已保存至 {chart_path}")
    return chart_path

def main_scraper(fund_code):
    """主函数，用于执行爬取和分析任务"""
    # 假设有一个 MySQL 数据库连接
    mysql = None  # PyMySQL(host='localhost', user='root', passwd='password', db='fund_db')
    mongodb = None # MongoClient('mongodb://localhost:27017/')

    print(f"开始获取基金 {fund_code} 的基本信息...")
    get_fund_details(fund_code, mysql=mysql)
    
    print(f"开始获取基金 {fund_code} 的历史净值...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    get_fund_data(fund_code, sdate=start_date, edate=end_date, mysql=mysql, mongodb=mongodb)
    
    print(f"开始使用 Selenium 获取基金 {fund_code} 的持仓数据...")
    output_filename = "fund_holdings.json"
    holdings_data = get_fund_holdings_with_selenium(fund_code)
    if holdings_data:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(holdings_data, f, indent=4, ensure_ascii=False)
        print(f"成功提取 {len(holdings_data)} 条持仓数据，并保存至 '{output_path}'。")
    
    print(f"开始获取基金 {fund_code} 的经理数据...")
    get_fund_managers(fund_code, mysql=mysql)
    
    print(f"开始分析基金 {fund_code} 的风险参数...")
    analysis_filename = "fund_analysis.json"
    risk_filename = "risk_metrics.json"
    analysis_result = analyze_fund(fund_code, start_date, end_date, use_yfinance=False, mongodb=mongodb)
    
    analysis_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), analysis_filename)
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=4, ensure_ascii=False)
    print(f"分析结果已保存至 '{analysis_path}'。")
    
    risk_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), risk_filename)
    risk_data = {
        "fund_code": fund_code,
        "annual_returns": analysis_result.get("annual_returns", None),
        "annual_volatility": analysis_result.get("annual_volatility", None),
        "sharpe_ratio": analysis_result.get("sharpe_ratio", None),
        "max_drawdown": analysis_result.get("max_drawdown", None)
    }
    with open(risk_path, 'w', encoding='utf-8') as f:
        json.dump(risk_data, f, indent=4, ensure_ascii=False)
    print(f"风险指标已保存至 '{risk_path}'。")

if __name__ == '__main__':
    fund_code_to_scrape = '023525'
    main_scraper(fund_code_to_scrape)
