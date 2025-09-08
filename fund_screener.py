import pandas as pd
import requests
import numpy as np
import json
import re
from datetime import datetime, timedelta
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 筛选条件
MIN_RETURN = 5.0  # 年化收益率 ≥ 5%
MAX_VOLATILITY = 20.0  # 年化波动率 ≤ 20%
MIN_SHARPE = 0.3  # 夏普比率 ≥ 0.3
MAX_FEE = 2.0  # 管理费 ≤ 2%
RISK_FREE_RATE = 3.0  # 无风险利率 3%

# 配置 requests 重试机制，以应对网络波动
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))

# --- 步骤 1: 获取基金列表 ---
def get_fund_list():
    """
    获取热门基金列表。
    """
    fund_list = [
        {'code': '161725', 'name': '招商中证白酒指数', 'type': '股票型'},
        {'code': '110011', 'name': '易方达中小盘混合', 'type': '混合型'},
        {'code': '510050', 'name': '华夏上证50ETF', 'type': '股票型'},
        {'code': '001593', 'name': '中欧医疗健康混合A', 'type': '混合型'},
        {'code': '519674', 'name': '银河创新成长混合', 'type': '混合型'}
    ]
    return pd.DataFrame(fund_list)

# --- 步骤 2: 获取历史净值（统一使用天天基金网的API）---
def get_fund_net_values(code, start_date, end_date):
    """
    从天天基金网的API获取基金历史净值数据。
    """
    print(f"尝试从天天基金网获取基金 {code} 历史净值...")
    url = f"http://api.fund.eastmoney.com/f10/lsjz?fundCode={code}&startDate={start_date}&endDate={end_date}&pageIndex=1&pageSize=50000"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
    }
    
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # 使用正则表达式从原始响应文本中提取JSON数据
        # 确保只解析大括号 {} 之间的内容
        json_string_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not json_string_match:
            print(f"获取基金 {code} 历史净值失败：响应内容中未找到有效的JSON结构。")
            return pd.DataFrame()

        json_string = json_string_match.group(0)

        # 尝试将提取出的JSON字符串解析为字典
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError:
            print(f"获取基金 {code} 历史净值失败：无法解析JSON数据。")
            print(f"原始响应内容前200字符：{response.text[:200]}...")
            return pd.DataFrame()

        if 'Data' not in data or not data['Data']['LSJZList']:
            print(f"获取基金 {code} 历史净值数据为空。")
            return pd.DataFrame()
            
        net_values = data['Data']['LSJZList']
        df = pd.DataFrame(net_values)
        
        # 重命名列并转换数据类型
        df = df.rename(columns={
            'FSRQ': 'date', 
            'DWJZ': 'net_value', 
            'LJJZ': 'total_value',
            'JZZZL': 'daily_return'
        })
        
        df['date'] = pd.to_datetime(df['date'])
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)
        
        print(f"基金 {code} 成功获取 {len(df)} 条净值数据。")
        return df

    except requests.exceptions.RequestException as e:
        print(f"获取基金 {code} 净值失败: {e}")
        return pd.DataFrame()

# --- 步骤 3: 获取管理费（统一使用天天基金网的API）---
def get_fund_fee(code):
    """
    从天天基金网的API获取基金管理费率。
    """
    print(f"尝试从天天基金网获取基金 {code} 管理费...")
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
    }
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 使用正则表达式从JavaScript代码中提取管理费率
        fee_match = re.search(r'data_fundTribble.ManagerFee=\'([\d.]+)\'', response.text)
        
        if fee_match:
            fee = float(fee_match.group(1))
            print(f"基金 {code} 管理费获取成功: {fee}%")
            return fee
        else:
            print(f"未在页面中找到基金 {code} 的管理费信息，使用默认值。")
            return 1.5 # 默认值

    except requests.exceptions.RequestException as e:
        print(f"获取基金 {code} 管理费失败: {e}")
        return 1.5 # 默认值

# --- 步骤 4: 计算指标 ---
def calculate_metrics(net_df):
    """
    计算基金的年化收益率、波动率和夏普比率。
    """
    if len(net_df) < 252: # 至少需要一年的数据（252个交易日）
        return None
        
    returns = net_df['net_value'].pct_change().dropna()
    
    # 累计收益率
    total_return = (net_df['net_value'].iloc[-1] / net_df['net_value'].iloc[0]) - 1
    
    # 年化收益率
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_return *= 100
    
    # 年化波动率
    volatility = returns.std() * np.sqrt(252) * 100
    
    # 夏普比率
    sharpe = (annual_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
    
    return {
        'annual_return': round(annual_return, 2),
        'volatility': round(volatility, 2),
        'sharpe': round(sharpe, 2)
    }

# --- 主函数 ---
def main():
    # 获取最近三年的数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime('%Y-%m-%d')

    funds_df = get_fund_list()
    if funds_df.empty:
        print("无法获取基金列表，程序退出。")
        return

    print(f"获取到 {len(funds_df)} 只候选基金。")
    results = []
    
    for idx, row in funds_df.iterrows():
        code = row['code']
        name = row['name']
        
        # 获取净值数据
        net_df = get_fund_net_values(code, start_date, end_date)
        if net_df.empty or len(net_df) < 252:
            print(f"跳过基金 {name}，数据不足以计算。")
            continue
            
        # 计算指标
        metrics = calculate_metrics(net_df)
        if metrics is None:
            print(f"跳过基金 {name}，计算指标失败。")
            continue

        # 获取管理费
        fee = get_fund_fee(code)
        
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
            }
            results.append(result)
            
        print("-" * 20)
        time.sleep(2)  # 避免触发反爬

    # 输出结果
    if results:
        final_df = pd.DataFrame(results).sort_values('年化收益率 (%)', ascending=False)
        print("\n--- 符合条件的推荐基金列表 ---")
        print(final_df)
        final_df.to_csv('recommended_funds.csv', index=False, encoding='utf-8-sig')
        print("\n结果已保存至 recommended_funds.csv 文件。")
    else:
        print("\n没有找到符合条件的基金，建议调整筛选条件。")

if __name__ == "__main__":
    main()
