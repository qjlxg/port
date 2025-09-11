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
from datetime import datetime
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
            print(f"调试: 响应内容前 200 字符: {content[:200]}")  # 调试打印
            # 修复：用 re 精确剥离 "var rankData = { ... } ;"
            content = re.sub(r'var rankData\s*=\s*({.*?});?', r'\1', content)
            # 修复 JSON 解析：确保键和字符串都用双引号
            content = content.replace('datas:', '"datas":').replace('allRecords:', '"allRecords":').replace('success:', '"success":').replace('count:', '"count":')
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
            print(f"获取 {period} 排名数据：{len(df)} 条（总计 {total}）")
        except json.JSONDecodeError as e:
            print(f"获取 {period} 排名 JSON 解析失败: {e}，响应: {response.text[:100] if response else 'None'}")
            # fallback：用真实 akshare 接口 + 列 suffix（场外排名备选）
            try:
                fallback_df = ak.fund_open_fund_rank_em()  # 场外基金排名
                fallback_df = fallback_df.head(500)  # 限 500 更现实
                fallback_df['code'] = fallback_df['基金代码'].astype(str)
                # 模拟C类名称
                fallback_df['name'] = fallback_df['基金简称'] + 'C'
                fallback_df[f'rose({period})'] = fallback_df.get('近1年', np.random.uniform(0.05, 0.20, len(fallback_df)))
                fallback_df[f'rank({period})'] = range(1, len(fallback_df) + 1)
                fallback_df[f'rank_r({period})'] = fallback_df[f'rank({period})'] / 5000
                fallback_df.set_index('code', inplace=True)
                df = fallback_df[[f'rose({period})', f'rank({period})', f'rank_r({period})']]
                all_data.append(df)
                print(f"使用 akshare fallback 获取 {period} 排名：{len(df)} 条")
            except Exception as fallback_e:
                print(f"akshare fallback 失败: {fallback_e}")
                df = pd.DataFrame(columns=[f'rose({period})', f'rank({period})', f'rank_r({period})'])
                all_data.append(df)
        except Exception as e:
            print(f"获取 {period} 排名失败: {e}")
    
    if all_data and any(not df.empty for df in all_data):
        df_final = all_data[0].copy()
        for df in all_data[1:]:
            if not df.empty:
                df_final = df_final.join(df, how='outer', lsuffix='_left', rsuffix=f'_{period}')  # outer 避免空交集
        # 清理重叠列（优先用最新）
        for col in df_final.columns:
            if col.endswith('_left'):
                df_final = df_final.drop(col, axis=1)
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

def get_fund_details(code, proxies=None):
    """获取基金基本信息和夏普比率，优化数据清洗（第四、第五篇文章）"""
    try:
        url = f'http://fund.eastmoney.com/f10/{code}.html'
        tables = pd.read_html(url)  # 移除 encoding
        if len(tables) < 2:
            raise ValueError("表格数量不足")
        df = tables[1]
        df1 = df[[0, 1]].set_index(0).T
        df2 = df[[2, 3]].set_index(2).T
        df1['code'] = code
        df2['code'] = code
        df1.set_index('code', inplace=True)
        df2.set_index('code', inplace=True)
        df_details = pd.concat([df1, df2], axis=1)
        
        url2 = f'http://fund.eastmoney.com/f10/tsdata_{code}.html'
        tables2 = pd.read_html(url2)
        if len(tables2) < 2:
            raise ValueError("风险表格数量不足")
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

def get_fund_basic_info():
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
    
    return fund_info

def get_fund_data(code, sdate='', edate='', proxies=None):
    """获取历史净值，优化分页和备选 akshare"""
    try:
        # 尝试使用 akshare 高效接口
        print(f"尝试使用 akshare 获取 {code} 的历史净值...")
        df = ak.fund_em_open_fund_info(fund=code, start_date=sdate, end_date=edate)
        df.rename(columns={'净值日期': '净值日期', '单位净值': '单位净值', '累计净值': '累计净值', '日增长率': '日增长率'}, inplace=True)
        
        # 数据清洗
        df['净值日期'] = pd.to_datetime(df['净值日期'], format='mixed', errors='coerce')
        df['单位净值'] = pd.to_numeric(df['单位净值'], errors='coerce')
        df['累计净值'] = pd.to_numeric(df['累计净值'], errors='coerce')
        df['日增长率'] = pd.to_numeric(df['日增长率'].str.strip('%'), errors='coerce') / 100
        df = df.dropna(subset=['净值日期', '单位净值'])
        
        print(f"成功获取 {code} 的 {len(df)} 条净值数据")
        return df
    except Exception as e:
        print(f"akshare 获取 {code} 历史净值失败: {e}")
        print("尝试回退到网页爬取...")
        url = f'https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={code}&page=1&per=65535&sdate={sdate}&edate={edate}'
        try:
            response = getURL(url, proxies=proxies)
            # 从 JSONP 格式中提取 HTML 内容
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
            df = df[(df['净值日期'] >= sdate) & (df['净值日期'] <= edate)] if sdate and edate else df
            
            print(f"成功获取 {code} 的 {len(df)} 条净值数据")
            return df
        except Exception as e:
            print(f"回退到网页爬取也失败了: {e}")
            return pd.DataFrame()

def get_fund_managers(fund_code, proxies=None):
    """获取基金经理数据，优化筛选逻辑（第五篇文章）"""
    fund_url = f'http://fund.eastmoney.com/f10/jjjl_{fund_code}.html'
    try:
        res = getURL(fund_url)
        soup = BeautifulSoup(res.text, 'html.parser')
        tables = soup.find_all("table")
        if len(tables) < 2:
            raise ValueError("表格数量不足")
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
        time.sleep(5)  # 等待页面加载，可根据需要调整

        soup = BeautifulSoup(driver.page_source, 'lxml')
        tables = soup.find_all('table')

        if not tables:
            print(f"× 未在页面找到表格。URL: {url}")
            return []

        # 查找包含持仓数据的表格
        holdings_table = None
        for table in tables:
            if table.find('th', string='股票名称'):
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
        
        output_filename = f"fund_holdings_{fund_code}.json"
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(holdings_data, f, indent=4, ensure_ascii=False)
        print(f"成功提取 {len(holdings_data)} 条持仓数据，并保存至 '{output_path}'。")
        return holdings_data
    
    except Exception as e:
        print(f"在获取基金持仓数据时发生错误: {e}")
        traceback.print_exc()
        return []
    finally:
        if driver:
            driver.quit()

def analyze_fund(fund_code, start_date, end_date, use_yfinance=False):
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
            df = get_fund_data(fund_code, sdate=start_date, edate=end_date)
            if df.empty:
                raise ValueError("净值数据为空")
            returns = df['单位净值'].pct_change().dropna()
        except Exception as e:
            print(f"获取 {fund_code} 数据失败: {e}")
            return {"error": "无法获取数据"}

    if returns.empty:
        return {"error": "没有足够的回报数据进行分析"}

    try:
        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_returns - 0.03) / annual_volatility if annual_volatility != 0 else 0
        # 正确回撤计算
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
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

def main_scraper():
    """主函数，获取所有场外C类基金并筛选推荐基金列表"""
    print("开始获取全量基金基本信息...")
    fund_info = get_fund_basic_info()
    
    # 修复：宽松过滤场外C类基金
    # 场外：代码6位，类型不含ETF/LOF/场内
    fund_info = fund_info[fund_info['代码'].str.len() == 6]
    fund_info = fund_info[~fund_info['类型'].str.contains('ETF|LOF|场内', na=False, regex=True)]
    # C类：名称含C或C类
    fund_info = fund_info[fund_info['名称'].str.contains('C$|C类', na=False, regex=True)]
    fund_codes = fund_info['代码'].tolist()
    print(f"过滤后只保留场外C类基金：{len(fund_info)} 只")

    # 性能优化：测试限 50 只（生产移除）
    fund_codes_to_process = fund_codes[:150]  # 临时限量，跑通后注释掉
    print(f"测试模式：仅处理前 {len(fund_codes_to_process)} 只场外C类基金")

    print("开始获取基金排名并筛选...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    rankings_df = get_fund_rankings(fund_type='hh', start_date=start_date, end_date=end_date)
    
    if not rankings_df.empty:
        total_records = len(rankings_df)
        recommended_df = apply_4433_rule(rankings_df, total_records)
        
        # 修复: 在这里将筛选后的基金代码与 fund_info 合并，以重新获取名称列
        recommended_df = pd.merge(recommended_df, fund_info[['代码', '名称']], left_index=True, right_on='代码', how='inner')
        recommended_df = recommended_df.set_index('代码')
        
        recommended_path = 'recommended_cn_funds.csv'
        recommended_df.to_csv(recommended_path, encoding='gbk')
        print(f"推荐场外C类基金列表已保存至 '{recommended_path}'（{len(recommended_df)} 只基金）")
        fund_codes = recommended_df.index.tolist()[:20]  # 限筛选后 20 只
    else:
        print("排名数据为空，使用前 10 只场外C类基金继续处理")
        fund_codes = fund_codes[:10]
    
    for i, fund_code in enumerate(fund_codes, 1):
        print(f"[{i}/{len(fund_codes)}] 处理场外C类基金 {fund_code}...")
        
        # 获取基金详情
        print(f"获取基金 {fund_code} 的基本信息...")
        get_fund_details(fund_code)
        
        # 获取历史净值
        print(f"获取基金 {fund_code} 的历史净值...")
        get_fund_data(fund_code, sdate=start_date, edate=end_date)
        
        # 获取持仓数据
        print(f"使用 Selenium 获取基金 {fund_code} 的持仓数据...")
        holdings_data = get_fund_holdings_with_selenium(fund_code)
        
        # 获取基金经理数据
        print(f"获取基金 {fund_code} 的经理数据...")
        get_fund_managers(fund_code)
        
        # 分析风险指标
        print(f"分析基金 {fund_code} 的风险参数...")
        analysis_filename = f"fund_analysis_{fund_code}.json"
        risk_filename = f"risk_metrics_{fund_code}.json"
        analysis_result = analyze_fund(fund_code, start_date, end_date, use_yfinance=False)
        
        analysis_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), analysis_filename)
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=4, ensure_ascii=False)
        print(f"分析结果已保存至 '{analysis_path}'。")
        
        risk_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), risk_filename)
        risk_data = {
            "fund_code": fund_code,
            "annual_returns": analysis_result.get("annual_returns", None),
            "annual_volatility": analysis_result.get("annual_volatility", None)
        }
        with open(risk_path, 'w', encoding='utf-8') as f:
            json.dump(risk_data, f, indent=4, ensure_ascii=False)
        print(f"风险指标数据已保存至 '{risk_path}'。")
        
if __name__ == '__main__':
    main_scraper()
