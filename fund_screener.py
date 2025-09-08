import akshare as ak
import requests
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 筛选条件（宽松）
MIN_RETURN = 5.0  # 年化收益率 ≥ 5%
MAX_VOLATILITY = 20.0  # 波动率 ≤ 20%
MIN_SHARPE = 0.3  # 夏普比率 ≥ 0.3
MAX_FEE = 2.0  # 管理费 ≤ 2%
RISK_FREE_RATE = 3.0  # 无风险利率 3%

# 配置 requests 重试机制
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))

# 步骤1: 获取基金列表（使用热门基金）
def get_fund_list():
    fund_list = [
        {'code': '161725', 'name': '招商中证白酒指数', 'type': '股票型'},
        {'code': '110011', 'name': '易方达中小盘混合', 'type': '混合型'},
        {'code': '510050', 'name': '华夏上证50ETF', 'type': '股票型'},
        {'code': '001593', 'name': '中欧医疗健康混合A', 'type': '混合型'},
        {'code': '519674', 'name': '银河创新成长混合', 'type': '混合型'}
    ]
    return pd.DataFrame(fund_list)

# 步骤2: 获取历史净值（AKShare 优先，失败则用天天基金网）
def get_fund_net_values(code, start_date, end_date):
    # AKShare 尝试
    try:
        net_df = ak.fund_open_fund_daily_em(symbol=code)  # 更新为正确接口
        net_df['日期'] = pd.to_datetime(net_df['日期'])
        net_df = net_df[(net_df['日期'] >= pd.to_datetime(start_date)) &
                        (net_df['日期'] <= pd.to_datetime(end_date))]
        net_df = net_df.sort_values('日期')
        ak_latest = net_df['单位净值'].iloc[-1] if not net_df.empty else None
        if len(net_df) >= 100:
            return net_df.rename(columns={'单位净值': 'net_value', '日期': 'date'}), ak_latest, 'AKShare'
        else:
            print(f"AKShare 获取基金 {code} 净值数据不足100天")
    except Exception as e:
        print(f"AKShare 获取基金 {code} 净值失败: {e}")
    
    # 回退到天天基金网
    url = f"http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={code}&page=1&sdate={start_date}&edate={end_date}&per=50000"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data_str = re.findall(r'data:(.*?),pages', response.text)
        if not data_str:
            print(f"天天基金网获取基金 {code} 净值数据为空")
            return pd.DataFrame(), None, 'None'
        net_values = json.loads(data_str[0])
        df = pd.DataFrame(net_values, columns=['date', 'net_value', 'total_value', 'daily_return', 'buy_status', 'sell_status'])
        df['date'] = pd.to_datetime(df['date'])
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df.sort_values('date').dropna(subset=['net_value'])
        tt_latest = df['net_value'].iloc[-1] if not df.empty else None
        if len(df) >= 100:
            return df, tt_latest, '天天基金网'
        else:
            print(f"天天基金网获取基金 {code} 净值数据不足100天")
            return pd.DataFrame(), None, 'None'
    except Exception as e:
        print(f"天天基金网获取基金 {code} 净值失败: {e}")
        return pd.DataFrame(), None, 'None'

# 步骤3: 获取管理费（天天基金网）
def get_fund_fee(code):
    # 手动指定热门基金管理费（从天天基金网查得）
    manual_fees = {
        '161725': 0.8,  # 招商中证白酒指数
        '110011': 1.5,  # 易方达中小盘混合
        '510050': 0.5,  # 华夏上证50ETF
        '001593': 1.5,  # 中欧医疗健康混合A
        '519674': 1.5   # 银河创新成长混合
    }
    if code in manual_fees:
        return manual_fees[code]
    
    url = f"http://fund.eastmoney.com/{code}.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        fee_match = re.search(r'管理费率</span>([\d.]+)%', response.text)
        return float(fee_match.group(1)) if fee_match else 1.0
    except Exception as e:
        print(f"获取基金 {code} 管理费失败: {e}")
        return 1.0

# 步骤4: 计算指标
def calculate_metrics(net_df):
    if len(net_df) < 100:
        return None
    returns = net_df['net_value'].pct_change().dropna()
    annual_return = (net_df['net_value'].iloc[-1] / net_df['net_value'].iloc[0]) ** (252 / len(returns)) - 1
    annual_return *= 100
    volatility = returns.std() * np.sqrt(252) * 100
    sharpe = (annual_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
    return {
        'annual_return': round(annual_return, 2),
        'volatility': round(volatility, 2),
        'sharpe': round(sharpe, 2)
    }

# 主函数
start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

funds_df = get_fund_list()
if funds_df.empty:
    print("无法获取基金列表，程序退出。")
    exit(1)

print(f"获取到 {len(funds_df)} 只候选基金")
results = []
debug_data = []
for idx, row in funds_df.iterrows():
    code = row['code']
    name = row['name']
    print(f"处理基金: {name} ({code})")
    # 获取净值
    net_df, latest_net_value, data_source = get_fund_net_values(code, start_date, end_date)
    if net_df.empty:
        debug_data.append({'基金代码': code, '基金名称': name, '失败原因': '无净值数据', '数据来源': data_source})
        continue
    # 计算指标
    metrics = calculate_metrics(net_df)
    if metrics is None:
        debug_data.append({'基金代码': code, '基金名称': name, '失败原因': '数据不足100天', '数据来源': data_source})
        continue
    # 获取管理费
    fee = get_fund_fee(code)
    # 调试信息
    debug_info = {
        '基金代码': code,
        '基金名称': name,
        '年化收益率 (%)': metrics['annual_return'],
        '波动率 (%)': metrics['volatility'],
        '夏普比率': metrics['sharpe'],
        '管理费 (%)': fee,
        '最新净值': latest_net_value,
        '数据来源': data_source
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
            '波动率 (%)': metrics['volatility'],
            '夏普比率': metrics['sharpe'],
            '管理费 (%)': round(fee, 2),
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
    time.sleep(2)  # 避免触发反爬

# 输出调试信息
debug_df = pd.DataFrame(debug_data)
debug_df.to_csv('debug_fund_metrics.csv', index=False, encoding='utf-8-sig')
print("调试信息保存到 debug_fund_metrics.csv")

# 输出结果
if results:
    final_df = pd.DataFrame(results).sort_values('综合评分', ascending=False)
    print("推荐的基金：")
    print(final_df)
    final_df.to_csv('recommended_cn_funds.csv', index=False, encoding='utf-8-sig')
    print("结果保存到 recommended_cn_funds.csv")
else:
    print("没有符合条件的基金，建议放宽筛选条件。")
