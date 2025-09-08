import requests
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
import time  # 避免请求太快被限

# 筛选条件（可调整）
MIN_RETURN = 7.0  # 最低年化收益率 (%)
MAX_VOLATILITY = 15.0  # 最大波动率 (%)
MIN_SHARPE = 0.5  # 最低夏普比率
MAX_FEE = 1.5  # 最高管理费 (%)
RISK_FREE_RATE = 3.0  # 无风险利率 (%)

# 步骤1: 获取所有基金列表
def get_all_funds():
    url = "http://fund.eastmoney.com/js/fundcode_search.js"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers)
    data_str = re.findall(r'\[.*\]', response.text)[0]
    funds = json.loads(data_str)
    df = pd.DataFrame(funds, columns=['code', 'pinyin', 'name', 'type', 'full_name'])
    # 过滤股票型和混合型基金（热门类型）
    stock_mixed = df[df['type'].isin(['股票型', '混合型'])]
    # 选取前 100 只热门基金（按代码排序，避免全抓太慢）
    return stock_mixed.head(100)

# 步骤2: 获取单基金历史净值（过去3年）
def get_fund_net_values(code, start_date, end_date):
    url = f"http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={code}&page=1&sdate={start_date}&edate={end_date}&per=50000"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers)
    data_str = re.findall(r'data:(.*?),pages', response.text)
    if not data_str:
        return pd.DataFrame()
    net_values = json.loads(data_str[0])
    df = pd.DataFrame(net_values, columns=['date', 'net_value', 'total_value', 'daily_return', 'buy_status', 'sell_status'])
    df['date'] = pd.to_datetime(df['date'])
    df['net_value'] = pd.to_numeric(df['net_value'])
    df = df.sort_values('date').dropna(subset=['net_value'])
    return df

# 步骤3: 获取基金管理费（从基本信息接口）
def get_fund_fee(code):
    url = f"http://fund.eastmoney.com/{code}.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers)
    # 简单提取管理费（实际可优化正则）
    fee_match = re.search(r'管理费率</span>([\d.]+)%', response.text)
    return float(fee_match.group(1)) if fee_match else 1.0  # 默认1.0%

# 步骤4: 计算指标
def calculate_metrics(net_df):
    if len(net_df) < 100:  # 至少100天数据
        return None
    returns = net_df['net_value'].pct_change().dropna()
    annual_return = (net_df['net_value'].iloc[-1] / net_df['net_value'].iloc[0]) ** (252 / len(returns)) - 1
    annual_return *= 100  # %
    volatility = returns.std() * np.sqrt(252) * 100  # %
    sharpe = (annual_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
    return {
        'annual_return': round(annual_return, 2),
        'volatility': round(volatility, 2),
        'sharpe': round(sharpe, 2)
    }

# 主函数
start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

funds_df = get_all_funds()
print(f"获取到 {len(funds_df)} 只候选基金")

results = []
for idx, row in funds_df.iterrows():
    code = row['code']
    print(f"处理基金: {row['name']} ({code})")
    net_df = get_fund_net_values(code, start_date, end_date)
    if net_df.empty:
        continue
    metrics = calculate_metrics(net_df)
    if metrics is None:
        continue
    fee = get_fund_fee(code)
    # 筛选
    if (metrics['annual_return'] >= MIN_RETURN and
        metrics['volatility'] <= MAX_VOLATILITY and
        metrics['sharpe'] >= MIN_SHARPE and
        fee <= MAX_FEE):
        result = {
            '基金代码': code,
            '基金名称': row['name'],
            '年化收益率 (%)': metrics['annual_return'],
            '波动率 (%)': metrics['volatility'],
            '夏普比率': metrics['sharpe'],
            '管理费 (%)': round(fee, 2)
        }
        # 综合评分
        score = (0.6 * (metrics['annual_return'] / 20) +  # 假设 max 20%
                 0.3 * metrics['sharpe'] +
                 0.1 * (2 - fee))  # 低费高分
        result['综合评分'] = round(score, 2)
        results.append(result)
    time.sleep(0.5)  # 延时，避免被限

# 输出结果
if results:
    final_df = pd.DataFrame(results).sort_values('综合评分', ascending=False)
    print("推荐的基金：")
    print(final_df)
    final_df.to_csv('recommended_cn_funds.csv', index=False, encoding='utf-8-sig')
    print("结果保存到 recommended_cn_funds.csv")
else:
    print("没有符合条件的基金，建议放宽筛选条件。")
