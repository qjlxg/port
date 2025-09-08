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

# 筛选条件（放宽）
MIN_RETURN = 3.0  # 年化收益率 ≥ 3%
MAX_VOLATILITY = 25.0  # 波动率 ≤ 25%
MIN_SHARPE = 0.2  # 夏普比率 ≥ 0.2
MAX_FEE = 2.5  # 管理费 ≤ 2.5%
RISK_FREE_RATE = 3.0  # 无风险利率 3%
MIN_DAYS = 100  # 最低数据天数（从 252 放宽到 100）

# 配置 requests 重试机制
session = requests.Session()
retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# 随机 User-Agent 列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

# 函数：获取基金净值历史数据
def get_fund_history_data(code):
    url = f'http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={code}&page=1&per=10000'
    headers = {'User-Agent': get_random_user_agent()}
    
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.text
        
        # 尝试从返回的HTML中提取数据
        try:
            pages = int(re.search(r'pages:(\d+),', html).group(1))
            total_records = int(re.search(r'records:(\d+),', html).group(1))
        except (AttributeError, ValueError):
            return None, "无法解析页数和记录数"

        url = f'http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={code}&page=1&per={total_records}'
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.text

        df = pd.read_html(html, header=0, encoding='utf-8')[0]
        
        # 检查数据是否完整
        if df.empty or '净值日期' not in df.columns or '日增长率' not in df.columns:
            return None, "数据列不完整"
        
        df = df.iloc[::-1]  # 倒序排列，日期从小到大
        df['净值日期'] = pd.to_datetime(df['净值日期'])
        df.set_index('净值日期', inplace=True)
        
        # 处理'日增长率'列，转换为数值类型
        df['日增长率'] = df['日增长率'].str.strip('%').astype(float)
        
        return df, None
    except requests.exceptions.RequestException as e:
        return None, f"请求数据失败: {e}"
    except ValueError as e:
        return None, f"解析HTML失败: {e}"

# 函数：获取基金实时估值和名称
def get_fund_realtime_data(code):
    url = f'http://fundgz.eastmoney.com/FundGuZhi/{code}.js'
    headers = {'User-Agent': get_random_user_agent()}
    
    try:
        response = session.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        # 提取JSON数据
        content = response.text
        json_str = re.search(r'jsonpgz\((.*?)\);', content).group(1)
        data = json.loads(json_str)
        
        # 获取名称和估值
        name = data.get('name', '未知')
        gszzl = data.get('gszzl', None)
        gsz = data.get('gsz', None)
        
        # 检查估值数据
        if gsz is not None and gszzl is not None:
            return name, float(gsz), float(gszzl)
        else:
            return name, None, None
    except requests.exceptions.RequestException as e:
        print(f"请求实时数据失败: {e}")
        return None, None, None
    except (AttributeError, json.JSONDecodeError) as e:
        print(f"解析实时数据失败: {e}")
        return None, None, None

# 函数：计算基金指标
def calculate_metrics(df, risk_free_rate):
    if df is None or len(df) < MIN_DAYS:
        return None, "数据天数不足"

    # 将日增长率转换为小数
    daily_returns = df['日增长率'] / 100
    
    # 计算年化收益率
    total_return = (1 + daily_returns).prod() - 1
    days = len(daily_returns)
    annual_return = ((1 + total_return) ** (365 / days) - 1) * 100

    # 计算年化波动率
    volatility = np.std(daily_returns) * np.sqrt(252) * 100

    # 计算夏普比率
    # 转换为日无风险利率
    daily_risk_free_rate = (1 + risk_free_rate / 100) ** (1/365) - 1
    sharpe = (daily_returns.mean() - daily_risk_free_rate) / daily_returns.std() * np.sqrt(252)
    
    metrics = {
        'annual_return': round(annual_return, 2),
        'volatility': round(volatility, 2),
        'sharpe': round(sharpe, 2),
    }
    return metrics, None

# 函数：获取基金管理费
def get_fund_fee(code):
    url = f'http://fund.eastmoney.com/{code}.html'
    headers = {'User-Agent': get_random_user_agent()}
    try:
        response = session.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        html = response.text
        
        # 使用正则表达式查找管理费
        match = re.search(r'管理费率：<span>(.*?)%</span>', html)
        if match:
            return float(match.group(1)), None
        
        return None, "未找到管理费"
    except requests.exceptions.RequestException as e:
        return None, f"请求管理费数据失败: {e}"
    except (AttributeError, ValueError) as e:
        return None, f"解析管理费失败: {e}"

# 函数：获取基金最新净值
def get_latest_net_value(code):
    url = f'http://fund.eastmoney.com/pingzhongdata/{code}.js'
    headers = {'User-Agent': get_random_user_agent()}
    try:
        response = session.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        js_content = response.text
        
        match = re.search(r'var Data_netWorthTrend = (.*?);', js_content)
        if match:
            net_worth_data = json.loads(match.group(1))
            if net_worth_data:
                latest_value = net_worth_data[-1]['y']
                return latest_value, None
        
        return None, "未找到最新净值"
    except requests.exceptions.RequestException as e:
        return None, f"请求最新净值数据失败: {e}"
    except (AttributeError, ValueError, json.JSONDecodeError) as e:
        return None, f"解析最新净值失败: {e}"

def get_fund_list():
    return [
        '005128', # 华夏上证50ETF联接A
        '460009', # 华泰柏瑞沪深300ETF联接A
        '004414', # 南方中证500ETF联接A
        '070010', # 嘉实沪深300ETF联接
        '110012', # 易方达深证100ETF联接
        '110037', # 易方达中证500ETF联接
        '100030', # 富国上证指数ETF联接
        '000051', # 华夏沪深300ETF联接A
        '110021', # 易方达上证中盘ETF联接
        '481005', # 工银瑞信央企ETF联接
        '519686', # 交银施罗德180公司治理联接
        '004453', # 万家上证50ETF联接
        '012362', # 国泰中证全指证券公司ETF联接A
        '000071', # 华夏恒生ETF联接
        '110026', # 易方达创业板ETF联接A
        '005967', # 国泰中证军工ETF联接A
        '180009', # 银华深证100联接
        '501057', # 汇添富中证新能源汽车产业指数(LOF)A
        '013013', # 华夏中证新能源汽车ETF发起式联接A
        '012717', # 易方达中证科技50ETF联接A
        '004488', # 嘉实富时中国A50ETF联接A
        '005545', # 中银中证银行ETF联接
        '100026', # 富国中证全指证券公司指数
        '159997', # 天弘中证电子ETF (无联接基金，保留原代码)
        '006096', # 广发消费ETF联接A
        '006098', # 华宝中证全指证券公司ETF联接
        '000551', # 华宝医疗ETF联接
        '012679', # 华泰柏瑞中证光伏产业ETF联接A
        '003756', # 易方达沪深300ETF联接A
        '004186', # 富国中证医药卫生ETF联接
        '008282', # 南方中证全指证券公司联接A
        '008889', # 华夏国证半导体芯片ETF联接A
        '007300', # 华夏国证半导体芯片ETF联接C
    ]

# 主程序
def main():
    fund_codes = get_fund_list()
    results = []
    debug_data = []

    for code in fund_codes:
        debug_info = {'基金代码': code}
        
        # 获取基金名称和实时估值
        name, realtime_estimate, _ = get_fund_realtime_data(code)
        debug_info['基金名称'] = name
        debug_info['实时估值'] = realtime_estimate
        
        print(f"处理基金：{code} ({name})")
        
        # 获取净值数据
        df_history, history_error = get_fund_history_data(code)
        if history_error:
            debug_info['失败原因'] = history_error
            debug_data.append(debug_info)
            print(f"跳过 {code}：{history_error}")
            continue
        
        # 计算指标
        metrics, metrics_error = calculate_metrics(df_history, RISK_FREE_RATE)
        if metrics_error:
            debug_info['失败原因'] = metrics_error
            debug_data.append(debug_info)
            print(f"跳过 {code}：{metrics_error}")
            continue

        # 获取管理费
        fee, fee_error = get_fund_fee(code)
        if fee_error:
            debug_info['失败原因'] = fee_error
            debug_data.append(debug_info)
            print(f"跳过 {code}：{fee_error}")
            continue

        latest_net_value, net_value_error = get_latest_net_value(code)
        if net_value_error:
            latest_net_value = 'N/A'
            print(f"获取最新净值失败：{net_value_error}")

        # 筛选和评分
        debug_info['年化收益率 (%)'] = metrics['annual_return']
        debug_info['年化波动率 (%)'] = metrics['volatility']
        debug_info['夏普比率'] = metrics['sharpe']
        debug_info['管理费 (%)'] = round(fee, 2)
        debug_info['最新净值'] = latest_net_value
        debug_info['数据来源'] = 'pingzhongdata'
        debug_info['数据条数'] = len(df_history)
        debug_info['数据开始日期'] = df_history.index.min().strftime('%Y-%m-%d')
        debug_info['数据结束日期'] = df_history.index.max().strftime('%Y-%m-%d')

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
                '数据来源': 'pingzhongdata'
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
        time.sleep(random.uniform(3, 7))  # 随机延时 3-7 秒

    # 保存调试信息
    debug_df = pd.DataFrame(debug_data)
    debug_df.to_csv('debug_fund_metrics.csv', index=False, encoding='utf-8-sig')
    print("调试信息已保存至 debug_fund_metrics.csv")

    # 输出结果
    if results:
        final_df = pd.DataFrame(results).sort_values('综合评分', ascending=False).reset_index(drop=True)
        final_df.index = final_df.index + 1
        print("筛选结果：")
        print(final_df.to_string())
        
        # 保存结果到 CSV
        final_df.to_csv('recommended_cn_funds.csv', index=False, encoding='utf-8-sig')
        print("最终结果已保存至 recommended_cn_funds.csv")
    else:
        print("没有找到符合筛选条件的基金。")

if __name__ == '__main__':
    main()
