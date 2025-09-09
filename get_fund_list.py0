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
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

def get_fund_list():
    """
    从天天基金网抓取场外基金信息（代码、名称、类型等），
    确保代码为6位数字，保存到 fund_codes.txt 和 fund_codes.csv。
    """
    print("正在从天天基金网获取场外基金列表，请稍候...")
    url = 'http://fund.eastmoney.com/js/fundcode_search.js'
    data = fetch_web_data(url)
    if not data:
        print("获取基金列表失败。")
        return False

    try:
        # 提取 JSON 数据
        json_str = re.search(r'var r = (\[.*?\]);', data, re.DOTALL).group(1)
        fund_list = json.loads(json_str)

        # 筛选场外基金：代码为6位数字，排除ETF、LOF、场内
        off_exchange_funds = [
            fund for fund in fund_list 
            if isinstance(fund[0], str) and re.match(r'^\d{6}$', fund[0]) 
            and isinstance(fund[2], str) and 'ETF' not in fund[2] 
            and 'LOF' not in fund[2] 
            and '场内' not in fund[2]
        ]

        if not off_exchange_funds:
            print("未找到任何场外基金。")
            return False

        # 保存6位基金代码到 fund_codes.txt
        with open('fund_codes.txt', 'w', encoding='utf-8') as f:
            for fund in off_exchange_funds:
                f.write(fund[0] + '\n')

        # 保存所有字段到 fund_codes.csv
        columns = ['代码', '简称', '类型', '拼音', '全称']
        df = pd.DataFrame(off_exchange_funds, columns=columns)
        df.to_csv('fund_codes.csv', index=False, encoding='utf-8-sig')

        print(f"成功获取 {len(off_exchange_funds)} 个场外基金信息，"
              f"已保存6位代码到 fund_codes.txt，所有字段到 fund_codes.csv。")
        return True
    except Exception as e:
        print(f"解析数据或保存文件时发生错误: {e}")
        return False

if __name__ == '__main__':
    get_fund_list()
