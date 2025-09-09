import pandas as pd
import requests
import json
import re
import random
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置 requests 重试机制
session = requests.Session()
retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# 随机 User-Agent 列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def fetch_web_data(url):
    """通用网页数据抓取函数，带随机UA和重试"""
    headers = {'User-Agent': get_random_user_agent()}
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text  # 返回文本内容
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

def get_fund_list():
    """
    从天天基金网抓取所有场外基金信息（代码、名称、类型等），并保存到 CSV 文件。
    """
    print("正在从天天基金网获取场外基金列表，请稍候...")
    # 天天基金网的基金数据 API
    url = 'http://fund.eastmoney.com/js/fundcode_search.js'
    data = fetch_web_data(url)
    if not data:
        print("获取基金列表失败。")
        return False

    try:
        # 提取 JSON 数据
        # API 返回的内容是一个 JavaScript 变量赋值语句，形如：var r = [...];
        json_str = re.search(r'var r = (\[.*?\]);', data).group(1)
        fund_list = json.loads(json_str)

        # 筛选场外基金：排除类型包含 'ETF'、'LOF' 或 '场内' 的基金
        off_exchange_funds = [
            fund for fund in fund_list 
            if re.match(r'^\d{6}$', fund[0])  # 确保代码是6位数字
            and 'ETF' not in fund[2] 
            and 'LOF' not in fund[2] 
            and '场内' not in fund[2]
        ]

        if not off_exchange_funds:
            print("未找到任何场外基金。")
            return False

        # 使用 pandas 保存所有字段到 CSV
        columns = ['代码', '简称', '类型', '拼音', '全称']
        df = pd.DataFrame(off_exchange_funds, columns=columns)
        df.to_csv('fund_codes.csv', index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 以支持 Excel 查看

        print(f"成功获取 {len(off_exchange_funds)} 个场外基金信息，并已保存到 fund_codes.csv 文件中。")
        return True
    except Exception as e:
        print(f"解析数据或保存文件时发生错误: {e}")
        return False

if __name__ == '__main__':
    get_fund_list()
