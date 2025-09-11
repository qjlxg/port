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
import random
from datetime import datetime, date, timedelta
import traceback

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

def get_realtime_valuation(fund_code):
    """获取实时估值"""
    try:
        url = f"http://fundgz.1234567.com.cn/js/{fund_code}.js"
        response = getURL(url)
        if response:
            data = json.loads(response.text.replace('jsonpgz(', '').replace(');', ''))
            realtime_price = float(data['dwjz'])
            return {
                'fund_code': fund_code,
                'realtime_price': realtime_price,
                'valuation_date': data.get('gztime', time.strftime('%Y-%m-%d %H:%M:%S'))
            }
    except Exception as e:
        print(f"天天基金实时估值失败 ({fund_code}): {e}")
    return None

def get_fund_rankings(fund_type='hh', proxies=None):
    """获取基金排名数据，优化分页处理"""
    today = date.today()
    periods = {
        '3y': (today - timedelta(days=365*3)).strftime('%Y-%m-%d'),
        '2y': (today - timedelta(days=365*2)).strftime('%Y-%m-%d'),
        '1y': (today - timedelta(days=365*1)).strftime('%Y-%m-%d'),
        '6m': (today - timedelta(days=30*6)).strftime('%Y-%m-%d'),
        '3m': (today - timedelta(days=30*3)).strftime('%Y-%m-%d')
    }
    all_data = []
    
    for period, sd in periods.items():
        url = f'http://fund.eastmoney.com/data/rankhandler.aspx?op=dy&dt=kf&ft={fund_type}&rs=&gs=0&sc=qjzf&st=desc&sd={sd}&ed={today.strftime("%Y-%m-%d")}&es=1&qdii=&pi=1&pn=10000&dx=1'
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
            df_final = df_final.join(df, how='inner', lsuffix=f'_{all_data.index(df)}', rsuffix=f'_{all_data.index(df)}')
        df_final.to_csv('fund_rankings.csv', encoding='gbk')
        print(f"排名数据已保存至 'fund_rankings.csv'")
        return df_final
    print("所有排名数据获取失败")
    return pd.DataFrame()

def apply_4433_rule(df):
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

def get_fund_details(code, proxies=None):
    """获取基金基本信息和夏普比率，优化数据清洗"""
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
        return df_final
    except Exception as e:
        print(f"获取基金 {code} 详情失败: {e}")
        return pd.DataFrame()

def get_fund_basic_info():
    """获取全量基金基本信息，添加 akshare 备选"""
    try:
        url = 'http://fund.eastmoney.com/js/fundcode_search.js'
        response = getURL(url)
        if response:
            text = re.findall(r'"(\d*?)","(.*?)","(.*?)","(.*?)","(.*?)"', response.text)
            fund_info_list = [{
                '代码': item[0], 
                '拼音简写': item[1], 
                '名称': item[2], 
                '类型': item[3], 
                '拼音全称': item[4]
            } for item in text]
            fund_info = pd.DataFrame(fund_info_list)
        else:
            print("天天基金基本信息获取失败，尝试 akshare...")
            fund_info = ak.fund_open_fund_info_em()
            fund_info = fund_info.rename(columns={'基金代码': '代码', '基金简称': '名称', '类型': '类型'})
    except Exception as e:
        print(f"获取基金基本信息失败: {e}")
        fund_info = ak.fund_open_fund_info_em()
        fund_info = fund_info.rename(columns={'基金代码': '代码', '基金简称': '名称', '类型': '类型'})
    
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
    fund_codes_df = pd.DataFrame({'基金代码': fund_info['代码'].tolist()})
    fund_codes_df.to_csv(codes_path, index=False, encoding='utf-8')
    print(f"全量基金代码列表已保存至 '{codes_path}'（{len(fund_info)} 只基金）")
    
    return fund_info

def get_fund_data(code, sdate='', edate='', proxies=None):
    """获取历史净值，优化分页和备选 akshare"""
    url = f'https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={code}&page=1&per=65535&sdate={sdate}&edate={edate}'
    try:
        response = getURL(url, proxies=proxies)
        content_match = re.search(r'content:"(.*?)"', response.text)
        if not content_match:
            raise ValueError("未找到净值表格内容")
        html_content = content_match.group(1).encode('utf-8').decode('unicode_escape')
        tree = etree.HTML(html_content)
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
        
        print(f"成功获取 {code} 的 {len(df)} 条净值数据")
        return df
    except Exception as e:
        print(f"lxml 解析失败 ({e})，尝试 akshare...")
        try:
            df = ak.fund_em_open_fund_daily(fund=code)
            if df.empty:
                raise ValueError("akshare 数据为空")
            df['净值日期'] = pd.to_datetime(df['净值日期'], format='mixed', errors='coerce')
            df = df[(df['净值日期'] >= sdate) & (df['净值日期'] <= edate)]
            print(f"akshare 获取 {code} 的 {len(df)} 条净值数据")
            return df
        except Exception as e:
            print(f"akshare 解析失败: {e}")
            return pd.DataFrame()

def get_fund_managers(fund_code):
    """获取基金经理数据，优化筛选逻辑"""
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
        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, 'lxml')
        tables = soup.find_all('table')

        if not tables:
            print(f"× 未在页面找到表格。URL: {url}")
            return []

        holdings_table = None
        for table in tables:
            if table.find('th', string='股票名称'):
                holdings_table = table
                break
        
        if not holdings_table:
            print(f"× 未在页面找到股票持仓表格。URL: {url}")
            return []

        holdings_data = []
        rows = holdings_table.find_all('tr')[1:]
        if not rows:
            print(f"× 股票持仓表格为空。URL: {url}")
            return []

        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 8:
                try:
                    stock_code = cols[1].text.strip()
                    stock_name = cols[2].text.strip()
                    proportion = cols[6].text.strip().replace('%', '')
                    shares = cols[7].text.strip()
                    value = cols[8].text.strip()

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
                    traceback.print_exc()
                    continue
        
        return holdings_data
    
    except Exception as e:
        print(f"在获取基金持仓数据时发生错误: {e}")
        traceback.print_exc()
        return []
    finally:
        if driver:
            driver.quit()

def analyze_fund(fund_code, start_date, end_date):
    """分析基金风险指标"""
    try:
        df = get_fund_data(fund_code, sdate=start_date, edate=end_date)
        returns = df['单位净值'].pct_change().dropna()
        if returns.empty:
            return {"error": "没有足够的回报数据进行分析"}

        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_returns - 0.03) / annual_volatility
        max_drawdown = (returns.max() - returns.min()) / returns.max()
        
        result = {
            "fund_code": fund_code,
            "annual_returns": float(annual_returns),
            "annual_volatility": float(annual_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown)
        }
        return result
    except Exception as e:
        print(f"分析基金 {fund_code} 风险参数失败: {e}")
        return {"error": "风险参数计算失败"}

def main_scraper():
    """主函数，用于执行全量基金筛选和分析任务"""
    
    # 1. 获取所有基金基本信息
    print("开始获取全量基金基本信息...")
    all_fund_info = get_fund_basic_info()
    if all_fund_info.empty:
        print("无法获取任何基金信息，退出。")
        return
        
    # 2. 获取基金排名数据并应用四四三三法则
    print("\n开始获取基金排名并应用四四三三法则...")
    ranking_df = get_fund_rankings(fund_type='hh')
    if ranking_df.empty:
        print("无法获取排名数据，退出。")
        return
        
    selected_funds = apply_4433_rule(ranking_df)
    if selected_funds.empty:
        print("四四三三法则未筛选出符合条件的基金，退出。")
        return
        
    selected_codes = selected_funds.index.tolist()
    
    all_analysis_results = []
    
    # 3. 对筛选出的基金进行深入分析
    print("\n开始对筛选出的基金进行深入分析...")
    for code in selected_codes:
        print(f"\n处理基金: {code}...")
        
        # 获取基本详情
        details_df = get_fund_details(code)
        
        # 获取基金经理数据
        managers_data = get_fund_managers(code)
        
        # 获取持仓数据
        holdings_data = get_fund_holdings_with_selenium(code)

        # 进行风险分析
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
        analysis_result = analyze_fund(code, start_date, end_date)
        
        # 合并所有数据
        fund_data = {
            'fund_code': code,
            'name': all_fund_info.loc[all_fund_info['代码'] == code, '名称'].iloc[0] if not all_fund_info[all_fund_info['代码'] == code].empty else 'N/A',
            'type': all_fund_info.loc[all_fund_info['代码'] == code, '类型'].iloc[0] if not all_fund_info[all_fund_info['代码'] == code].empty else 'N/A',
            'details': details_df.to_dict('records') if not details_df.empty else {},
            'managers': managers_data,
            'holdings': holdings_data,
            'analysis': analysis_result,
            'ranking': selected_funds.loc[code].to_dict() if code in selected_funds.index else {}
        }
        all_analysis_results.append(fund_data)
        
    # 4. 保存最终结果
    print("\n所有筛选和分析任务已完成，开始保存最终结果...")
    
    # 保存所有分析结果到 JSON
    analysis_filename = "fund_analysis.json"
    analysis_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), analysis_filename)
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(all_analysis_results, f, indent=4, ensure_ascii=False)
    print(f"所有分析结果已保存至 '{analysis_path}'。")

    # 保存推荐基金列表到 CSV
    recommended_funds_list = []
    for fund in all_analysis_results:
        rec_data = {
            'fund_code': fund['fund_code'],
            'name': fund['name'],
            'type': fund['type'],
            '3y_ranking': fund['ranking'].get('rank_r(3y)', 'N/A'),
            '2y_ranking': fund['ranking'].get('rank_r(2y)', 'N/A'),
            '1y_ranking': fund['ranking'].get('rank_r(1y)', 'N/A'),
            '6m_ranking': fund['ranking'].get('rank_r(6m)', 'N/A'),
            '3m_ranking': fund['ranking'].get('rank_r(3m)', 'N/A'),
            'sharpe_ratio': fund['analysis'].get('sharpe_ratio', 'N/A'),
            'annual_returns': fund['analysis'].get('annual_returns', 'N/A'),
            'max_drawdown': fund['analysis'].get('max_drawdown', 'N/A'),
            'managers': ', '.join([m['fund_managers'] for m in fund['managers']])
        }
        recommended_funds_list.append(rec_data)
    
    recommended_df = pd.DataFrame(recommended_funds_list)
    recommended_csv_path = 'recommended_cn_funds.csv'
    recommended_df.to_csv(recommended_csv_path, index=False, encoding='utf-8')
    print(f"最终推荐基金列表已保存至 '{recommended_csv_path}'。")

if __name__ == '__main__':
    main_scraper()
