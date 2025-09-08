import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import warnings

# 忽略 pandas 在解析 HTML 时可能出现的警告
warnings.filterwarnings('ignore', category=UserWarning)

def download_and_parse_data():
    """
    使用 requests 和 BeautifulSoup 从天天基金网下载并解析基金数据。
    这个版本会更智能地找到目标表格，并精确提取数据。
    """
    url = 'http://fund.eastmoney.com/fund.html'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print("正在尝试从天天基金网下载数据...")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = 'gbk'

        soup = BeautifulSoup(response.text, 'html.parser')

        # 查找页面上所有可能的表格
        tables = soup.find_all('table')
        
        target_table = None
        for table in tables:
            # 查找表头行，并检查是否包含关键列名
            header_row = table.find('tr', class_='h')
            if header_row and '基金代码' in header_row.get_text() and '基金简称' in header_row.get_text():
                target_table = table
                break
        
        if not target_table:
            print("错误：未找到包含'基金代码'和'基金简称'的表格。网站结构可能已变化。")
            return None

        # 提取数据行
        data_rows = target_table.find_all('tr', id=re.compile('^tr'))
        if not data_rows:
            print("错误：未找到任何数据行。")
            return None

        # 确定列数以避免长度不匹配错误
        num_cols = len(data_rows[0].find_all('td'))
        
        data = []
        for row in data_rows:
            row_data = [re.sub(r'\s+', '', td.get_text(strip=True)) for td in row.find_all('td')]
            if len(row_data) == num_cols:
                data.append(row_data)
        
        df = pd.DataFrame(data)

        # 固定列名，确保一致性
        df.columns = ['关注', '比较', '序号', '基金代码', '基金简称', '单位净值', '累计净值',
                      '昨日单位净值', '昨日累计净值', '日增长值', '日增长率', '申购状态', '赎回状态', '手续费']
        
        print("\n数据下载成功，已识别以下列：")
        print(df.columns.tolist())

        # 转换数据类型并进行清理
        df['日增长率'] = pd.to_numeric(df['日增长率'].astype(str).str.rstrip('%').replace('--', '0'), errors='coerce') / 100
        
        return df

    except requests.exceptions.RequestException as e:
        print(f"发生网络请求错误：{e}")
        return None
    except Exception as e:
        print(f"发生解析错误：{e}")
        return None

def screen_funds(df):
    if df is None or df.empty:
        return None

    print("\n开始执行基金筛选...")
    min_daily_growth_rate = 0.05
    filtered_funds = df[df['日增长率'] > min_daily_growth_rate].copy()
    return filtered_funds

if __name__ == "__main__":
    fund_data_df = download_and_parse_data()
    if fund_data_df is not None:
        qualified_funds = screen_funds(fund_data_df)
        
        print("\n--- 符合筛选条件的基金列表 ---")
        if qualified_funds is not None and not qualified_funds.empty:
            print(qualified_funds[['基金代码', '基金简称', '日增长率']])
            qualified_funds.to_csv('qualified_funds.csv', index=False, encoding='utf_8_sig')
            print("\n结果已保存至 qualified_funds.csv 文件。")
        else:
            print("没有找到符合条件的基金。请尝试调整筛选条件。")
