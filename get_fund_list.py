import requests
import json
import re
from tqdm import tqdm

def get_fund_list():
    """
    从天天基金网获取所有场内基金代码列表
    """
    url = "http://fund.eastmoney.com/js/fundcode_search.js"
    try:
        print("正在获取所有基金代码，请稍候...", flush=True)
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # 使用正则表达式提取 JSON 数据
        match = re.search(r'var r = (\[.*?\]);', response.text)
        if not match:
            print("错误：无法从网页内容中解析基金代码数据。", flush=True)
            return []
        
        data = json.loads(match.group(1))

        # 筛选出场内基金（代码长度为6位）
        fund_codes = [item[0] for item in data if len(item[0]) == 6]
        
        print(f"成功获取 {len(fund_codes)} 个基金代码。", flush=True)
        return fund_codes

    except requests.exceptions.RequestException as e:
        print(f"错误：获取基金代码列表失败，请检查网络连接。{e}", flush=True)
        return []

def save_to_file(fund_codes, file_path='fund_codes.txt'):
    """
    将基金代码列表保存到文件
    """
    if not fund_codes:
        print("没有可保存的基金代码。", flush=True)
        return
        
    with open(file_path, 'w', encoding='utf-8') as f:
        for code in tqdm(fund_codes, desc="保存中"):
            f.write(code + '\n')
    print(f"基金代码已保存至 {file_path}", flush=True)

if __name__ == '__main__':
    all_funds = get_fund_list()
    save_to_file(all_funds)
