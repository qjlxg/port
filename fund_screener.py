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

# 筛选条件（可根据你的需求进行调整）
MIN_RETURN = 3.0  # 年化收益率 ≥ 3%
MAX_VOLATILITY = 25.0  # 波动率 ≤ 25%
MIN_SHARPE = 0.2  # 夏普比率 ≥ 0.2
MAX_FEE = 2.5  # 管理费 ≤ 2.5%
RISK_FREE_RATE = 3.0  # 无风险利率 3%
MIN_DAYS = 100  # 最低数据天数

# 配置 requests 重试机制
session = requests.Session()
retries = Retry(total=5, backoff_factor=2, status_for_list=[429, 500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# 随机 User-Agent 列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

# 步骤 1: 获取基金列表
def get_fund_list():
    """
    获取热门基金列表，覆盖多种类型，用于筛选。
    """
    fund_list = [
        # 宽基指数基金 (Broad-based Index Funds)
        {'code': '510050', 'name': '华夏上证50ETF', 'type': '指数型'},
        {'code': '510300', 'name': '华泰柏瑞沪深300ETF', 'type': '指数型'},
        {'code': '510500', 'name': '南方中证500ETF', 'type': '指数型'},
        {'code': '159919', 'name': '嘉实沪深300ETF', 'type': '指数型'},
        {'code': '159905', 'name': '易方达深证100ETF', 'type': '指数型'},
        {'code': '510880', 'name': '易方达中证500ETF', 'type': '指数型'},
        {'code': '510210', 'name': '富国上证指数ETF', 'type': '指数型'},
        {'code': '510330', 'name': '华夏沪深300ETF', 'type': '指数型'},
        {'code': '510130', 'name': '易方达上证中盘ETF', 'type': '指数型'},
        {'code': '510060', 'name': '工银瑞信央企ETF', 'type': '指数型'},
        {'code': '510010', 'name': '交银施罗德180治理ETF', 'type': '指数型'},
        {'code': '510680', 'name': '万家上证50ETF基金', 'type': '指数型'},
        {'code': '512880', 'name': '国泰中证全指证券公司ETF', 'type': '股票型'},
        {'code': '513520', 'name': '华夏恒生ETF', 'type': '股票型'},
        {'code': '159915', 'name': '易方达创业板ETF', 'type': '股票型'},
        {'code': '512690', 'name': '国泰中证军工ETF', 'type': '股票型'},
        {'code': '159995', 'name': '银华深证100ETF', 'type': '股票型'},

        # 行业主题基金 (Sector-specific Funds)
        {'code': '161725', 'name': '招商中证白酒指数', 'type': '股票型'},
        {'code': '001593', 'name': '中欧医疗健康混合A', 'type': '混合型'},
        {'code': '005827', 'name': '易方达蓝筹精选混合', 'type': '混合型'},
        {'code': '501057', 'name': '汇添富中证新能源ETF', 'type': '股票型'},
        {'code': '005911', 'name': '广发双擎升级混合A', 'type': '混合型'},
        {'code': '006751', 'name': '嘉实农业产业股票', 'type': '股票型'},
        {'code': '001103', 'name': '工银瑞信前沿医疗股票', 'type': '股票型'},
        {'code': '003986', 'name': '前海开源公用事业股票', 'type': '股票型'},
        {'code': '001633', 'name': '工银瑞信文体产业股票', 'type': '股票型'},
        {'code': '160629', 'name': '博时医疗保健行业混合', 'type': '混合型'},
        {'code': '007300', 'name': '南方中证全指证券公司ETF', 'type': '股票型'},
        {'code': '004734', 'name': '中欧时代先锋股票A', 'type': '股票型'},
        {'code': '001607', 'name': '景顺长城新兴成长混合', 'type': '混合型'},
        {'code': '005501', 'name': '中泰星元灵活配置混合', 'type': '混合型'},
        {'code': '005615', 'name': '东方红睿元三年定期混合', 'type': '混合型'},
        {'code': '000041', 'name': '华夏回报二号混合', 'type': '混合型'},
        {'code': '001210', 'name': '华夏行业景气混合', 'type': '混合型'},
        {'code': '159842', 'name': '华夏中证新能源车ETF', 'type': '股票型'},
        {'code': '515050', 'name': '中证科技50ETF', 'type': '股票型'},
        {'code': '513100', 'name': '中金金瑞富时A50ETF', 'type': '股票型'},
        {'code': '512980', 'name': '中银中证银行ETF', 'type': '股票型'},
        {'code': '512760', 'name': '华夏国证半导体芯片ETF', 'type': '股票型'},
        {'code': '512290', 'name': '富国证券ETF', 'type': '股票型'},
        {'code': '159997', 'name': '华宝医疗ETF', 'type': '股票型'},
        {'code': '159937', 'name': '广发消费ETF', 'type': '股票型'},
        {'code': '512800', 'name': '华宝券商ETF', 'type': '股票型'},
        {'code': '512170', 'name': '华泰柏瑞光伏ETF', 'type': '股票型'},

        # 绩优基金 (High-Performing Funds)
        {'code': '003095', 'name': '富国天惠精选成长混合', 'type': '混合型'},
        {'code': '110011', 'name': '易方达中小盘混合', 'type': '混合型'},
        {'code': '519674', 'name': '银河创新成长混合', 'type': '混合型'},
        {'code': '000001', 'name': '华夏成长混合', 'type': '混合型'},
        {'code': '000002', 'name': '华夏回报混合A', 'type': '混合型'},
        {'code': '000003', 'name': '华夏回报混合B', 'type': '混合型'},
        {'code': '000004', 'name': '华夏兴华混合', 'type': '混合型'},
        {'code': '000005', 'name': '华夏行业混合', 'type': '混合型'},
        {'code': '000006', 'name': '华夏优势混合', 'type': '混合型'},
        {'code': '000011', 'name': '华夏大盘精选混合', 'type': '混合型'},
        {'code': '000961', 'name': '华夏盛世混合', 'type': '混合型'},
        {'code': '001016', 'name': '工银瑞信新金融股票A', 'type': '股票型'},
        {'code': '001052', 'name': '中银策略精选混合', 'type': '混合型'},
        {'code': '001103', 'name': '工银瑞信前沿医疗股票', 'type': '股票型'},
        {'code': '001186', 'name': '广发稳健增长混合A', 'type': '混合型'},
        {'code': '001550', 'name': '嘉实新兴产业股票', 'type': '股票型'},
        {'code': '001607', 'name': '景顺长城新兴成长混合', 'type': '混合型'},
        {'code': '001633', 'name': '工银瑞信文体产业股票', 'type': '股票型'},
        {'code': '001700', 'name': '易方达新兴成长混合', 'type': '混合型'},
        {'code': '001878', 'name': '汇添富创新成长混合', 'type': '混合型'},
        {'code': '002011', 'name': '招商核心价值混合', 'type': '混合型'},
        {'code': '002014', 'name': '广发医疗保健股票A', 'type': '股票型'},
        {'code': '002120', 'name': '博时新兴成长混合', 'type': '混合型'},
        {'code': '002131', 'name': '富国新动力混合A', 'type': '混合型'},
        {'code': '002157', 'name': '广发新经济混合', 'type': '混合型'},
        {'code': '002206', 'name': '华安媒体互联网混合', 'type': '混合型'},
        {'code': '002213', 'name': '银华富裕主题混合', 'type': '混合型'},
        {'code': '002220', 'name': '嘉实新收益混合', 'type': '混合型'},
        {'code': '002251', 'name': '兴全精选混合', 'type': '混合型'},
        {'code': '002324', 'name': '鹏华新兴产业混合', 'type': '混合型'},
        {'code': '002359', 'name': '博时主题行业混合', 'type': '混合型'},
        {'code': '002410', 'name': '中欧新蓝筹混合A', 'type': '混合型'},
        {'code': '002574', 'name': '广发聚丰混合', 'type': '混合型'},
        {'code': '002621', 'name': '国泰金马稳健回报混合', 'type': '混合型'},
        {'code': '002711', 'name': '易方达消费行业股票', 'type': '股票型'},
        {'code': '002888', 'name': '富国天惠成长混合', 'type': '混合型'},
        {'code': '002939', 'name': '中欧价值发现混合A', 'type': '混合型'},
        {'code': '003003', 'name': '兴全合润混合', 'type': '混合型'},
        {'code': '003095', 'name': '富国天惠精选成长混合', 'type': '混合型'},
        {'code': '003294', 'name': '东方红睿华沪港深混合', 'type': '混合型'},
        {'code': '003463', 'name': '兴全社会责任混合', 'type': '混合型'},
        {'code': '003634', 'name': '广发创新升级混合', 'type': '混合型'},
        {'code': '003756', 'name': '易方达中小板ETF', 'type': '股票型'},
        {'code': '003767', 'name': '国泰量化收益混合', 'type': '混合型'},
        {'code': '004077', 'name': '工银瑞信创新动力混合', 'type': '混合型'},
        {'code': '004186', 'name': '华泰柏瑞沪深300ETF联接A', 'type': '指数型'},
        {'code': '004241', 'name': '华夏移动互联混合', 'type': '混合型'},
        {'code': '004314', 'name': '中欧明睿新常态混合', 'type': '混合型'},
        {'code': '004357', 'name': '易方达消费行业股票A', 'type': '股票型'},
        {'code': '004423', 'name': '富国低碳新经济混合', 'type': '混合型'},
        {'code': '004513', 'name': '招商中证白酒指数C', 'type': '股票型'},
        {'code': '004658', 'name': '广发中证全指医药卫生ETF联接A', 'type': '指数型'},
        {'code': '004734', 'name': '中欧时代先锋股票A', 'type': '股票型'},
        {'code': '004751', 'name': '富国新天地产混合', 'type': '混合型'},
        {'code': '004851', 'name': '易方达创新驱动混合', 'type': '混合型'},
        {'code': '004944', 'name': '广发新兴成长混合A', 'type': '混合型'},
        {'code': '005063', 'name': '中欧消费主题股票A', 'type': '股票型'},
        {'code': '005064', 'name': '中欧医疗健康混合C', 'type': '混合型'},
        {'code': '005273', 'name': '富国新材料新能源混合', 'type': '混合型'},
        {'code': '005312', 'name': '广发多元新兴股票', 'type': '股票型'},
        {'code': '005470', 'name': '嘉实智能汽车股票', 'type': '股票型'},
        {'code': '005501', 'name': '中泰星元灵活配置混合', 'type': '混合型'},
        {'code': '005615', 'name': '东方红睿元三年定期混合', 'type': '混合型'},
        {'code': '005662', 'name': '富国消费主题混合', 'type': '混合型'},
        {'code': '005814', 'name': '广发高端制造股票A', 'type': '股票型'},
        {'code': '005827', 'name': '易方达蓝筹精选混合', 'type': '混合型'},
        {'code': '005863', 'name': '嘉实成长精选混合', 'type': '混合型'},
        {'code': '005911', 'name': '广发双擎升级混合A', 'type': '混合型'},
        {'code': '005937', 'name': '易方达价值精选股票', 'type': '股票型'},
        {'code': '006152', 'name': '富国创新成长混合', 'type': '混合型'},
        {'code': '006228', 'name': '景顺长城内需增长混合', 'type': '混合型'},
        {'code': '006751', 'name': '嘉实农业产业股票', 'type': '股票型'},
        {'code': '006764', 'name': '招商中证白酒指数C', 'type': '股票型'},
        {'code': '006935', 'name': '国泰智能汽车股票', 'type': '股票型'},
        {'code': '007300', 'name': '南方中证全指证券公司ETF', 'type': '股票型'},
        {'code': '007577', 'name': '富国成长精选混合', 'type': '混合型'},
        {'code': '007622', 'name': '易方达战略新兴产业混合', 'type': '混合型'},
        {'code': '007804', 'name': '中欧阿尔法混合A', 'type': '混合型'},
        {'code': '007810', 'name': '兴全合宜混合(LOF)', 'type': '混合型'},
        {'code': '008282', 'name': '富国中证医药卫生ETF', 'type': '股票型'},
        {'code': '008514', 'name': '易方达消费精选股票', 'type': '股票型'},
        {'code': '008889', 'name': '易方达沪深300ETF联接A', 'type': '指数型'}
    ]
    return pd.DataFrame(fund_list)

# 步骤 2: 获取历史净值（主备双重保险）
def get_fund_net_values(code, start_date, end_date):
    """
    尝试从两个接口获取基金历史净值数据。
    """
    print(f"\n尝试获取基金 {code} 历史净值...")
    # 尝试主接口 (pingzhongdata)
    df, latest_value = get_net_values_from_pingzhongdata(code, start_date, end_date)
    if not df.empty and len(df) >= MIN_DAYS:
        return df, latest_value, 'pingzhongdata'

    # 如果主接口失败，尝试备用接口 (lsjz)
    print(f"主接口获取失败，尝试备用接口 (lsjz)...")
    df, latest_value = get_net_values_from_lsjz(code, start_date, end_date)
    if not df.empty and len(df) >= MIN_DAYS:
        return df, latest_value, 'lsjz'

    # 如果两个接口都失败，返回空
    print(f"基金 {code} 数据获取失败")
    return pd.DataFrame(), None, 'None'

def get_net_values_from_pingzhongdata(code, start_date, end_date):
    """从 fund.eastmoney.com/pingzhongdata/ 接口获取净值"""
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Host': 'fund.eastmoney.com',
        'Accept': 'application/javascript, */*;q=0.8'
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        print(f"pingzhongdata 响应状态码: {response.status_code}")
        response.raise_for_status()

        if not response.text.strip():
            print(f"pingzhongdata 获取基金 {code} 失败：响应内容为空")
            return pd.DataFrame(), None

        net_worth_match = re.search(r'Data_netWorthTrend\s*=\s*(\[.*?\]);', response.text, re.DOTALL)
        if not net_worth_match:
            print(f"pingzhongdata 获取基金 {code} 失败：无法找到 Data_netWorthTrend")
            return pd.DataFrame(), None

        try:
            net_worth_list = json.loads(net_worth_match.group(1))
        except json.JSONDecodeError as e:
            print(f"pingzhongdata 获取基金 {code} 失败：JSON 解析错误，{e}")
            return pd.DataFrame(), None

        df = pd.DataFrame(net_worth_list)
        if 'x' not in df or 'y' not in df:
            print(f"pingzhongdata 获取基金 {code} 失败：数据缺少 x 或 y 字段")
            return pd.DataFrame(), None

        df = df.rename(columns={'x': 'date', 'y': 'net_value'})
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df[(df['date'] >= pd.to_datetime(start_date)) &
                (df['date'] <= pd.to_datetime(end_date))]
        df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)

        if df.empty:
            print(f"pingzhongdata 获取基金 {code} 失败：日期范围内无数据")
            return pd.DataFrame(), None

        latest_value = df['net_value'].iloc[-1] if not df.empty else None
        print(f"pingzhongdata 基金 {code} 获取 {len(df)} 条数据，范围：{df['date'].iloc[0]} 至 {df['date'].iloc[-1]}")
        return df, latest_value

    except requests.exceptions.RequestException as e:
        print(f"pingzhongdata 获取基金 {code} 失败: {e}")
        return pd.DataFrame(), None

def get_net_values_from_lsjz(code, start_date, end_date):
    """从 fund.eastmoney.com/f10/lsjz 接口获取净值"""
    url = f"http://fund.eastmoney.com/f10/lsjz?fundCode={code}&pageIndex=1&pageSize=50000"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/f10/fjcc_{code}.html',
        'Host': 'fund.eastmoney.com',
        'Accept': 'application/json, text/javascript, */*; q=0.01'
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        print(f"lsjz 响应状态码: {response.status_code}")
        response.raise_for_status()

        data_str_match = re.search(r'var\s+apidata=\{content:"(.*?)",', response.text, re.DOTALL)
        if not data_str_match:
            print(f"lsjz 获取基金 {code} 失败：无法找到 apidata.content")
            return pd.DataFrame(), None

        json_data_str = data_str_match.group(1).replace("\\", "")
        try:
            data = json.loads(json_data_str)
        except json.JSONDecodeError as e:
            print(f"lsjz 获取基金 {code} 失败：JSON 解析错误，{e}")
            return pd.DataFrame(), None

        if 'LSJZList' not in data or not data['LSJZList']:
            print(f"lsjz 获取基金 {code} 失败：LSJZList 为空")
            return pd.DataFrame(), None

        df = pd.DataFrame(data['LSJZList'])
        df = df.rename(columns={'FSRQ': 'date', 'DWJZ': 'net_value', 'LJJZ': 'total_value', 'JZZZL': 'daily_return'})
        df['date'] = pd.to_datetime(df['date'])
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df[(df['date'] >= pd.to_datetime(start_date)) &
                (df['date'] <= pd.to_datetime(end_date))]
        df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)

        if df.empty:
            print(f"lsjz 获取基金 {code} 失败：日期范围内无数据")
            return pd.DataFrame(), None

        latest_value = df['net_value'].iloc[-1] if not df.empty else None
        print(f"lsjz 基金 {code} 获取 {len(df)} 条数据，范围：{df['date'].iloc[0]} 至 {df['date'].iloc[-1]}")
        return df, latest_value

    except requests.exceptions.RequestException as e:
        print(f"lsjz 获取基金 {code} 失败: {e}")
        return pd.DataFrame(), None

# 步骤 3: 获取实时估值
def get_fund_realtime_estimate(code):
    """
    从 fundgz.1234567.com.cn 接口获取基金实时估值。
    """
    print(f"尝试获取基金 {code} 实时估值...")
    url = f"http://fundgz.1234567.com.cn/js/{code}.js?rt={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Host': 'fundgz.1234567.com.cn'
    }
    try:
        response = session.get(url, headers=headers, timeout=10)
        print(f"实时估值响应状态码: {response.status_code}")
        response.raise_for_status()
        match = re.search(r'jsonpgz\((.*)\)', response.text, re.DOTALL)
        if match:
            json_data = json.loads(match.group(1))
            gsz = json_data.get('gsz')
            print(f"基金 {code} 实时估值: {gsz}")
            return float(gsz) if gsz else None
        print(f"获取基金 {code} 实时估值失败：无法解析 JSONP")
        return None
    except Exception as e:
        print(f"获取基金 {code} 实时估值失败: {e}")
        return None

# 步骤 4: 获取管理费
def get_fund_fee(code):
    """
    从 fund.eastmoney.com/pingzhongdata/ 接口获取管理费，带手动备用。
    """
    manual_fees = {
        '510050': 0.5, '510300': 0.5, '510500': 0.5, '159919': 0.5, '159905': 0.5,
        '510880': 0.5, '510210': 0.5, '510330': 0.5, '510130': 0.5, '510060': 0.5,
        '510010': 0.5, '510680': 0.5, '512880': 0.5, '513520': 0.5, '159915': 0.5,
        '512690': 0.5, '159995': 0.5, '161725': 0.8, '001593': 1.5, '005827': 1.5,
        '501057': 0.5, '005911': 1.5, '006751': 1.5, '001103': 1.5, '003986': 1.5,
        '001633': 1.5, '160629': 1.5, '007300': 0.5, '004734': 1.5, '001607': 1.5,
        '005501': 1.5, '005615': 1.5, '000041': 1.5, '001210': 1.5, '159842': 0.5,
        '515050': 0.5, '513100': 0.5, '512980': 0.5, '512760': 0.5, '512290': 0.5,
        '159997': 0.5, '159937': 0.5, '512800': 0.5, '512170': 0.5, '003095': 1.5,
        '110011': 1.5, '519674': 1.5, '000001': 1.5, '000002': 1.5, '000003': 1.5,
        '000004': 1.5, '000005': 1.5, '000006': 1.5, '000011': 1.5, '000961': 1.5,
        '001016': 1.5, '001052': 1.5, '001186': 1.5, '001550': 1.5, '001700': 1.5,
        '001878': 1.5, '002011': 1.5, '002014': 1.5, '002120': 1.5, '002131': 1.5,
        '002157': 1.5, '002206': 1.5, '002213': 1.5, '002220': 1.5, '002251': 1.5,
        '002324': 1.5, '002359': 1.5, '002410': 1.5, '002574': 1.5, '002621': 1.5,
        '002711': 1.5, '002888': 1.5, '002939': 1.5, '003003': 1.5, '003294': 1.5,
        '003463': 1.5, '003634': 1.5, '003756': 0.5, '003767': 1.5, '004077': 1.5,
        '004186': 0.5, '004241': 1.5, '004314': 1.5, '004357': 1.5, '004423': 1.5,
        '004513': 0.8, '004658': 0.5, '004751': 1.5, '004851': 1.5, '004944': 1.5,
        '005063': 1.5, '005064': 1.5, '005273': 1.5, '005312': 1.5, '005470': 1.5,
        '005662': 1.5, '005814': 1.5, '005863': 1.5, '005937': 1.5, '006152': 1.5,
        '006228': 1.5, '006764': 0.8, '006935': 1.5, '007577': 1.5, '007622': 1.5,
        '007804': 1.5, '007810': 1.5, '008282': 0.5, '008514': 1.5, '008889': 0.5
    }
    if code in manual_fees:
        print(f"基金 {code} 使用手动管理费: {manual_fees[code]}%")
        return manual_fees[code]

    print(f"尝试从天天基金网获取基金 {code} 管理费...")
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Host': 'fund.eastmoney.com'
    }
    try:
        response = session.get(url, headers=headers, timeout=10)
        print(f"管理费响应状态码: {response.status_code}")
        response.raise_for_status()
        fee_match = re.search(r'data_fundTribble\.ManagerFee=\'([\d.]+)\'', response.text)
        if fee_match:
            fee = float(fee_match.group(1))
            print(f"基金 {code} 管理费获取成功: {fee}%")
            return fee
        print(f"未在页面中找到基金 {code} 的管理费信息，使用默认值 1.5%")
        return 1.5
    except requests.exceptions.RequestException as e:
        print(f"获取基金 {code} 管理费失败: {e}")
        return 1.5

# 步骤 5: 获取基金持仓信息
def get_fund_holdings(code):
    """
    从天天基金网获取基金持仓信息。
    """
    print(f"尝试获取基金 {code} 的持仓详情...")
    url = f"http://fund.eastmoney.com/DataCenter/Fund/JJZCHoldDetail.aspx?fundCode={code}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/f10/jjcc_{code}.html'
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        holdings = []
        stock_table = soup.find('table', {'class': 'm-table'})
        if stock_table:
            for row in stock_table.find_all('tr')[1:]:
                cells = row.find_all('td')
                if len(cells) >= 4:
                    holdings.append({
                        'name': cells[1].text.strip(),
                        'code': cells[2].text.strip(),
                        'ratio': cells[3].text.strip()
                    })
        return holdings
    except requests.exceptions.RequestException as e:
        print(f"获取基金 {code} 持仓信息失败: {e}")
        return []
    except Exception as e:
        print(f"解析基金 {code} 持仓信息时出错: {e}")
        return []

# 步骤 6: 计算指标
def calculate_metrics(net_df, start_date, end_date):
    """
    计算基金的年化收益率、波动率和夏普比率。
    """
    net_df = net_df[(net_df['date'] >= pd.to_datetime(start_date)) &
                    (net_df['date'] <= pd.to_datetime(end_date))].copy()
    
    if len(net_df) < MIN_DAYS:
        print(f"数据不足 {MIN_DAYS} 天，仅有 {len(net_df)} 天")
        return None
        
    returns = net_df['net_value'].pct_change().dropna()
    total_return = (net_df['net_value'].iloc[-1] / net_df['net_value'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_return *= 100
    volatility = returns.std() * np.sqrt(252) * 100
    sharpe = (annual_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
    return {
        'annual_return': round(annual_return, 2),
        'volatility': round(volatility, 2),
        'sharpe': round(sharpe, 2)
    }

# 主函数
def main():
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime('%Y-%m-%d')

    funds_df = get_fund_list()
    if funds_df.empty:
        print("无法获取基金列表，程序退出。")
        return

    print(f"获取到 {len(funds_df)} 只候选基金。")
    results = []
    debug_data = []

    for idx, row in funds_df.iterrows():
        code = row['code']
        name = row['name']
        print(f"\n处理基金: {name} ({code})")

        # 获取历史净值数据
        net_df, latest_net_value, data_source = get_fund_net_values(code, start_date, end_date)
        
        if net_df.empty:
            debug_data.append({
                '基金代码': code,
                '基金名称': name,
                '失败原因': '无净值数据',
                '数据来源': data_source
            })
            continue

        # 计算指标
        metrics = calculate_metrics(net_df, start_date, end_date)
        if metrics is None:
            debug_data.append({
                '基金代码': code,
                '基金名称': name,
                '失败原因': f'数据不足 {MIN_DAYS} 天（仅有 {len(net_df)} 天）',
                '数据来源': data_source,
                '数据条数': len(net_df)
            })
            continue

        # 获取管理费
        fee = get_fund_fee(code)
        
        # 获取实时估值
        realtime_estimate = get_fund_realtime_estimate(code)

        # 调试信息
        debug_info = {
            '基金代码': code,
            '基金名称': name,
            '年化收益率 (%)': metrics['annual_return'],
            '年化波动率 (%)': metrics['volatility'],
            '夏普比率': metrics['sharpe'],
            '管理费 (%)': fee,
            '最新净值': latest_net_value,
            '实时估值': realtime_estimate,
            '数据来源': data_source,
            '数据条数': len(net_df),
            '数据开始日期': net_df['date'].iloc[0].strftime('%Y-%m-%d') if not net_df.empty else None,
            '数据结束日期': net_df['date'].iloc[-1].strftime('%Y-%m-%d') if not net_df.empty else None
        }

        # 筛选
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
            results.append(result)
        else:
            debug_info['失败原因'] = '未通过筛选'
        debug_data.append(debug_info)
        print("-" * 50)
        time.sleep(random.uniform(3, 7))

    # 保存调试信息
    debug_df = pd.DataFrame(debug_data)
    debug_df.to_csv('debug_fund_metrics.csv', index=False, encoding='utf-8-sig')
    print("调试信息已保存至 debug_fund_metrics.csv")

    # 输出结果并获取持仓信息
    if results:
        final_df = pd.DataFrame(results).sort_values('综合评分', ascending=False)
        print("\n--- 符合条件的推荐基金列表 ---")
        print(final_df)
        final_df.to_csv('recommended_cn_funds.csv', index=False, encoding='utf-8-sig')
        print("\n结果已保存至 recommended_cn_funds.csv 文件。")

        # ------------------- 新增功能：获取持仓信息 -------------------
        for idx, row in final_df.iterrows():
            code = row['基金代码']
            name = row['基金名称']
            
            holdings = get_fund_holdings(code)
            
            if holdings:
                print(f"\n--- 基金 {name} ({code}) 持仓详情 ---")
                for holding in holdings:
                    print(f"  - 股票名称: {holding.get('name', 'N/A')}, 股票代码: {holding.get('code', 'N/A')}, 占比: {holding.get('ratio', 'N/A')}")
            else:
                print(f"\n--- 基金 {name} ({code}) 未能获取到持仓数据 ---")
            
            print("-" * 50)
            time.sleep(random.uniform(2, 5))
            
    else:
        print("\n没有找到符合条件的基金，建议调整筛选条件。")

if __name__ == "__main__":
    main()
