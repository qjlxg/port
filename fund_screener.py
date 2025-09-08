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

# --- 步骤 2: 获取历史净值 ---
def get_fund_net_values(code, start_date, end_date):
    """
    从天天基金网 API 获取基金历史净值数据。
    """
    print(f"尝试从天天基金网获取基金 {code} 历史净值...")
    # 使用更稳定的接口，返回的是包含 JSON 的 HTML
    url = f"http://fund.eastmoney.com/f10/lsjz?fundCode={code}&pageIndex=1&pageSize=50000"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': f'http://fund.eastmoney.com/f10/fjcc_{code}.html',
        'Host': 'fund.eastmoney.com'
    }
    
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # 预处理：使用正则表达式从响应内容中提取 JSON 字符串
        data_str_match = re.search(r'var\s+apidata=\{content:"(.*?)",', response.text, re.DOTALL)
        if not data_str_match:
            print(f"获取基金 {code} 历史净值失败：无法在响应中找到数据。")
            return pd.DataFrame(), None, '天天基金网'
            
        # 移除转义字符
        json_data_str = data_str_match.group(1).replace("\\", "")

        # 解析 JSON
        try:
            data = json.loads(json_data_str)
        except json.JSONDecodeError as e:
            print(f"获取基金 {code} 历史净值失败：响应内容不是有效 JSON，错误：{e}")
            return pd.DataFrame(), None, '天天基金网'
        
        if 'LSJZList' not in data or not data['LSJZList']:
            print(f"获取基金 {code} 历史净值数据为空或格式错误")
            return pd.DataFrame(), None, '天天基金网'
            
        net_values = data['LSJZList']
        df = pd.DataFrame(net_values)
        df = df.rename(columns={'FSRQ': 'date', 'DWJZ': 'net_value', 'LJJZ': 'total_value', 'JZZZL': 'daily_return'})
        df['date'] = pd.to_datetime(df['date'])
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)
        latest_net_value = df['net_value'].iloc[-1] if not df.empty else None
        print(f"基金 {code} 成功获取 {len(df)} 条净值数据。")
        return df, latest_net_value, '天天基金网'
        
    except requests.exceptions.RequestException as e:
        print(f"获取基金 {code} 净值失败: {e}")
        return pd.DataFrame(), None, '天天基金网'

# --- 步骤 3: 获取管理费 ---
def get_fund_fee(code):
    """
    从天天基金网获取管理费，带手动备用。
    """
    manual_fees = {
        '161725': 0.8,  # 招商中证白酒指数
        '110011': 1.5,  # 易方达中小盘混合
        '510050': 0.5,  # 华夏上证50ETF
        '001593': 1.5,  # 中欧医疗健康混合A
        '519674': 1.5,  # 银河创新成长混合
        '501057': 0.5,  # 汇添富中证新能源ETF
        '005911': 1.5,  # 广发双擎升级混合A
        '006751': 1.5    # 嘉实农业产业股票
    }
    if code in manual_fees:
        print(f"基金 {code} 使用手动管理费: {manual_fees[code]}%")
        return manual_fees[code]

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
            print(f"基金 {code} 管理费获取成功: {fee}%")
            return fee
        else:
            print(f"未在页面中找到基金 {code} 的管理费信息，使用默认值 1.5%")
            return 1.5
    except requests.exceptions.RequestException as e:
        print(f"获取基金 {code} 管理费失败: {e}")
        return 1.5

# --- 步骤 4: 计算指标 ---
def calculate_metrics(net_df):
    """
    计算基金的年化收益率、波动率和夏普比率。
    """
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
    debug_data = []

    for idx, row in funds_df.iterrows():
        code = row['code']
        name = row['name']
        print(f"\n处理基金: {name} ({code})")

        # 获取净值数据
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
        metrics = calculate_metrics(net_df)
        if metrics is None:
            debug_data.append({
                '基金代码': code,
                '基金名称': name,
                '失败原因': f'数据不足252天（仅有 {len(net_df)} 天）',
                '数据来源': data_source
            })
            continue

        # 获取管理费
        fee = get_fund_fee(code)

        # 调试信息
        debug_info = {
            '基金代码': code,
            '基金名称': name,
            '年化收益率 (%)': metrics['annual_return'],
            '年化波动率 (%)': metrics['volatility'],
            '夏普比率': metrics['sharpe'],
            '管理费 (%)': fee,
            '最新净值': latest_net_value,
            '数据来源': data_source,
            '数据条数': len(net_df)
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
                '数据来源': data_source
            }
            # 计算综合评分
            score = (0.6 * (metrics['annual_return'] / 20) +
                     0.3 * metrics['sharpe'] +
                     0.1 * (2 - fee))
            result['综合评分'] = round(score, 2)
            results.append(result)
        else:
            debug_info['失败原因'] = '未通过筛选'
        debug_data.append(debug_info)
        print("-" * 50)
        time.sleep(random.uniform(2, 5))  # 随机延时 2-5 秒

    # 保存调试信息
    debug_df = pd.DataFrame(debug_data)
    debug_df.to_csv('debug_fund_metrics.csv', index=False, encoding='utf-8-sig')
    print("调试信息已保存至 debug_fund_metrics.csv")

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
