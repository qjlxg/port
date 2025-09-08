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

# 筛选条件（宽松）
MIN_RETURN = 5.0  # 年化收益率 ≥ 5%
MAX_VOLATILITY = 20.0  # 波动率 ≤ 20%
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
    获取热门基金列表，覆盖多种类型。
    """
    fund_list = [
        {'code': '161725', 'name': '招商中证白酒指数', 'type': '股票型'},
        {'code': '110011', 'name': '易方达中小盘混合', 'type': '混合型'},
        {'code': '510050', 'name': '华夏上证50ETF', 'type': '股票型'},
        {'code': '001593', 'name': '中欧医疗健康混合A', 'type': '混合型'},
        {'code': '519674', 'name': '银河创新成长混合', 'type': '混合型'},
        {'code': '501057', 'name': '汇添富中证新能源ETF', 'type': '股票型'},
        {'code': '005911', 'name': '广发双擎升级混合A', 'type': '混合型'},
        {'code': '006751', 'name': '嘉实农业产业股票', 'type': '股票型'}
    ]
    return pd.DataFrame(fund_list)

# --- 步骤 2: 获取历史净值（主备双重保险）---
def get_fund_net_values(code):
    """
    尝试从两个不同接口获取基金历史净值数据。
    """
    # 尝试主接口 (pingzhongdata)
    df, latest_value = get_net_values_from_pingzhongdata(code)
    if not df.empty and len(df) > 252:
        return df, latest_value, 'pingzhongdata'

    # 如果主接口失败，尝试备用接口 (lsjz)
    print("主接口获取失败，尝试备用接口...")
    df, latest_value = get_net_values_from_lsjz(code)
    if not df.empty and len(df) > 252:
        return df, latest_value, 'lsjz'

    # 如果两个接口都失败，返回空
    return pd.DataFrame(), None, 'None'

def get_net_values_from_pingzhongdata(code):
    """从 fund.eastmoney.com/pingzhongdata/ 接口获取净值"""
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Host': 'fund.eastmoney.com'
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        net_worth_match = re.search(r'Data_netWorthTrend\s*=\s*(\[.*?\]);', response.text, re.DOTALL)
        if not net_worth_match:
            return pd.DataFrame(), None
            
        # 替换单引号为双引号以符合 JSON 格式
        net_worth_json_str = net_worth_match.group(1).replace("'", '"')
        net_worth_list = json.loads(net_worth_json_str)
        
        df = pd.DataFrame(net_worth_list, columns=['date', 'net_value', 'daily_return'])
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)
        latest_value = df['net_value'].iloc[-1] if not df.empty else None
        print(f"基金 {code} 从 pingzhongdata 接口获取 {len(df)} 条数据。")
        return df, latest_value
    except requests.exceptions.RequestException as e:
        print(f"pingzhongdata 接口获取失败: {e}")
        return pd.DataFrame(), None
    except Exception as e:
        print(f"pingzhongdata 接口解析失败: {e}")
        return pd.DataFrame(), None

def get_net_values_from_lsjz(code):
    """从 fund.eastmoney.com/f10/lsjz 接口获取净值"""
    url = f"http://fund.eastmoney.com/f10/lsjz?fundCode={code}&pageIndex=1&pageSize=50000"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': f'http://fund.eastmoney.com/f10/fjcc_{code}.html',
        'Host': 'fund.eastmoney.com'
    }
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        data_str_match = re.search(r'var\s+apidata=\{content:"(.*?)",', response.text, re.DOTALL)
        if not data_str_match:
            return pd.DataFrame(), None
            
        json_data_str = data_str_match.group(1).replace("\\", "")
        data = json.loads(json_data_str)
        
        if 'LSJZList' not in data or not data['LSJZList']:
            return pd.DataFrame(), None
            
        df = pd.DataFrame(data['LSJZList'])
        df = df.rename(columns={'FSRQ': 'date', 'DWJZ': 'net_value', 'LJJZ': 'total_value', 'JZZZL': 'daily_return'})
        df['date'] = pd.to_datetime(df['date'])
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)
        latest_value = df['net_value'].iloc[-1] if not df.empty else None
        print(f"基金 {code} 从 lsjz 接口获取 {len(df)} 条数据。")
        return df, latest_value
    except requests.exceptions.RequestException as e:
        print(f"lsjz 接口获取失败: {e}")
        return pd.DataFrame(), None
    except Exception as e:
        print(f"lsjz 接口解析失败: {e}")
        return pd.DataFrame(), None

# --- 步骤 3: 获取实时估值 ---
def get_fund_realtime_estimate(code):
    """
    从 fundgz.1234567.com.cn 接口获取基金实时估值。
    """
    print(f"尝试获取基金 {code} 实时估值...")
    url = f"http://fundgz.1234567.com.cn/js/{code}.js?rt={int(time.time() * 1000)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)',
    }
    
    try:
        response = session.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        match = re.search(r'jsonpgz\((.*)\)', response.text, re.DOTALL)
        if match:
            json_data = json.loads(match.group(1))
            gsz = json_data.get('gsz')
            return float(gsz) if gsz else None
        return None
    except Exception as e:
        print(f"获取基金 {code} 实时估值失败: {e}")
        return None

# --- 步骤 4: 获取管理费 ---
def get_fund_fee(code):
    """
    从 fund.eastmoney.com/pingzhongdata/ 接口获取基金管理费。
    """
    print(f"尝试从天天基金网获取基金 {code} 管理费...")
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Host': 'fund.eastmoney.com'
    }
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        fee_match = re.search(r'data_fundTribble\.ManagerFee=\'([\d.]+)\'', response.text)
        if fee_match:
            fee = float(fee_match.group(1))
            return fee
        else:
            return 1.5
    except requests.exceptions.RequestException as e:
        print(f"获取基金 {code} 管理费失败: {e}")
        return 1.5

# --- 步骤 5: 计算指标 ---
def calculate_metrics(net_df, start_date, end_date):
    """
    计算基金的年化收益率、波动率和夏普比率。
    """
    # 筛选出指定日期范围内的数据
    net_df = net_df[(net_df['date'] >= start_date) & (net_df['date'] <= end_date)].copy()
    
    if len(net_df) < 252:
        print(f"数据不足252天，仅有 {len(net_df)} 天")
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

# --- 主函数 ---
def main():
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
        print(f"\n处理基金: {name} ({code})")

        # 获取历史净值数据（自动选择最优接口）
        net_df, latest_net_value, data_source = get_fund_net_values(code)
        
        if net_df.empty:
            print(f"跳过基金 {name}，数据获取失败。")
            continue
            
        # 计算指标
        metrics = calculate_metrics(net_df, start_date, end_date)
        if metrics is None:
            print(f"跳过基金 {name}，数据不足以计算。")
            continue

        # 获取管理费
        fee = get_fund_fee(code)
        
        # 获取实时估值
        realtime_estimate = get_fund_realtime_estimate(code)

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
                '实时估值 (最新)': round(realtime_estimate, 4) if realtime_estimate is not None else 'N/A',
                '数据来源': data_source
            }
            # 计算综合评分
            score = (0.6 * (metrics['annual_return'] / 20) +
                     0.3 * metrics['sharpe'] +
                     0.1 * (2 - fee))
            result['综合评分'] = round(score, 2)
            results.append(result)
        
        print("-" * 50)
        time.sleep(random.uniform(2, 5))  # 随机延时 2-5 秒

    # 输出结果
    if results:
        final_df = pd.DataFrame(results).sort_values('综合评分', ascending=False)
        print("\n--- 符合条件的推荐基金列表 ---")
        print(final_df)
        final_df.to_csv('recommended_cn_funds.csv', index=False, encoding='utf-8-sig')
        print("\n结果已保存至 recommended_cn_funds.csv 文件。")
    else:
        print("\n没有找到符合条件的基金，建议调整筛选条件。")

if __name__ == "__main__":
    main()
