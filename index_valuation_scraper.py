# 导入所需的库
import requests
import json
import csv
import os
from datetime import datetime

# 请求头，模拟浏览器访问
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36'
}

# 估值指数API的URL
INDEX_API_URL = 'https://danjuanfunds.com/djapi/index_eva/dj'

def get_index_data():
    """
    从丹juan基金API获取指数估值数据。
    """
    try:
        response = requests.get(INDEX_API_URL, headers=header, timeout=10)
        # 如果HTTP请求返回失败状态码，则抛出异常
        response.raise_for_status()
        
        data = response.json()
        
        # 检查API响应是否包含我们所需的数据
        if data.get('data') and data['data'].get('items'):
            return data['data']['items']
        else:
            print("错误：API返回的数据结构不正确或无数据。")
            return None
    except requests.exceptions.RequestException as e:
        print(f"请求指数估值数据失败：{e}")
        return None
    except json.JSONDecodeError:
        print("无法解析API响应，返回的不是有效的JSON。")
        return None

def comprehensive_filter_indices(data):
    """
    综合筛选出估值低且基本面优秀的指数。
    """
    selected_indices = []
    
    # PE百分位低于20% 且 股息率大于3%
    pe_percentile_threshold = 0.20
    yeild_threshold = 0.03
    
    for item in data:
        pe_percentile = item.get('pe_percentile')
        yeild = item.get('yeild')
        
        # 检查关键字段是否存在且符合筛选条件
        if pe_percentile is not None and yeild is not None:
            if pe_percentile < pe_percentile_threshold and yeild > yeild_threshold:
                selected_indices.append(item)
    
    return selected_indices

def save_to_csv(data, filename):
    """
    将数据保存到CSV文件。
    """
    if not data:
        print(f"没有数据可保存到 {filename}。")
        return
        
    today_date = datetime.now().strftime('%Y-%m-%d')
    csv_file_path = os.path.join(os.getcwd(), filename)
    
    # CSV文件的表头
    fieldnames = ['日期', '指数名称', '指数代码', 'PE', 'PE百分位', 'PB', 'PB百分位', '股息率', 'ROE']
    
    try:
        # 使用 'w' 模式写入文件，如果文件存在则覆盖，确保每次运行都是最新的数据
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 写入数据
            for item in data:
                row = {
                    '日期': today_date,
                    '指数名称': item.get('name', ''),
                    '指数代码': item.get('index_code', ''),
                    'PE': item.get('pe', ''),
                    'PE百分位': item.get('pe_percentile', ''),
                    'PB': item.get('pb', ''),
                    'PB百分位': item.get('pb_percentile', ''),
                    '股息率': item.get('yeild', ''),
                    'ROE': item.get('roe', '')
                }
                writer.writerow(row)
        print(f"数据已成功保存到 {csv_file_path}")
    except IOError as e:
        print(f"写入文件时出错：{e}")

if __name__ == '__main__':
    print("开始获取指数估值数据...")
    index_data = get_index_data()
    
    if index_data:
        # 1. 保存所有指数的估值数据
        save_to_csv(index_data, filename='all_index_valuation.csv')
        
        # 2. 综合筛选出低估且基本面好的指数
        selected_data = comprehensive_filter_indices(index_data)
        
        if selected_data:
            print("\n根据综合筛选策略，发现以下值得关注的指数：")
            
            # 将筛选结果保存到单独的文件
            save_to_csv(selected_data, filename='selected_funds.csv')

            for index in selected_data:
                print(f"- {index.get('name')} (PE百分位: {index.get('pe_percentile')*100:.2f}%, PB百分位: {index.get('pb_percentile')*100:.2f}%)")
