import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import warnings

# 忽略 pandas 在解析 HTML 时可能出现的警告
warnings.filterwarnings('ignore', category=UserWarning)

# --- 步骤 1: 自动下载基金数据 (使用更智能的BeautifulSoup) ---
def download_fund_data():
    """
    使用 requests 和 BeautifulSoup 从天天基金网下载并解析基金数据。
    这个版本会更智能地找到目标表格。
    """
    url = 'http://fund.eastmoney.com/fund.html'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print("正在尝试从天天基金网下载数据...")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # 如果请求失败，抛出异常
        response.encoding = 'gbk'

        soup = BeautifulSoup(response.text, 'html.parser')

        # 查找页面中的所有表格
        tables = soup.find_all('table')

        target_table = None
        for table in tables:
            # 检查表格中是否有包含 '基金代码' 和 '基金简称' 的表头
            header_row = table.find('tr', class_='h')
            if header_row and '基金代码' in header_row.get_text() and '基金简称' in header_row.get_text():
                target_table = table
                break
        
        if not target_table:
            print("错误：未找到目标基金数据表格。请检查网站结构是否变化。")
            return None

        # 提取表头
        headers_row = target_table.find('tr', class_='h')
        headers_texts = [re.sub(r'\s+', '', th.get_text(strip=True)) for th in headers_row.find_all('th')]

        # 由于网页表头是两层的，我们需要找到所有列名
        # 一个更可靠的方法是直接从数据行提取td数量来推断列数
        first_data_row = target_table.find('tr', id=re.compile('^tr'))
        num_cols = len(first_data_row.find_all('td')) if first_data_row else 0
        if num_cols == 0:
            print("错误：表格中没有找到数据行。")
            return None
        
        # 提取所有数据行
        data = []
        for row in target_table.find_all('tr', id=re.compile('^tr')):
            row_data = [re.sub(r'\s+', '', td.get_text(strip=True)) for td in row.find_all('td')]
            if len(row_data) == num_cols:
                data.append(row_data)

        df = pd.DataFrame(data)

        # 打印原始列名以供调试
        print("以下是下载数据中找到的原始列名:")
        print(headers_texts)

        # 使用一个可靠的列名列表来重命名列
        df.columns = ['关注', '比较', '序号', '基金代码', '基金简称', '单位净值', '累计净值',
                      '昨日单位净值', '昨日累计净值', '日增长值', '日增长率', '申购状态', '赎回状态', '手续费']

        # 筛选出我们需要的列
        df = df[['基金代码', '基金简称', '日增长率']]
        
        # 转换数据类型
        df['日增长率'] = pd.to_numeric(df['日增长率'].astype(str).str.rstrip('%').replace('--', '0'), errors='coerce') / 100

        print("\n注意: 当前页面不包含所有筛选所需的数据 (如近1年/3年收益率)。")
        print("此版本脚本仅能基于现有数据进行处理。")
        
        return df

    except requests.exceptions.RequestException as e:
        print(f"发生网络请求错误：{e}")
        print("请检查网络连接或确认URL是否正确。")
        return None
    except Exception as e:
        print(f"发生解析错误：{e}")
        return None

# --- 步骤 2: 设定筛选条件并执行筛选 ---
def screen_funds(df):
    """
    根据预设条件筛选基金。
    """
    if df is None or df.empty:
        return None

    print("\n开始执行基金筛选...")

    min_daily_growth_rate = 0.05  # 日增长率 > 5%

    filtered_funds = df[df['日增长率'] > min_daily_growth_rate].copy()

    return filtered_funds

# --- 步骤 3: 运行主程序 ---
if __name__ == "__main__":
    fund_data_df = download_fund_data()
    
    if fund_data_df is not None:
        qualified_funds = screen_funds(fund_data_df)
        
        print("\n--- 符合筛选条件的基金列表 ---")
        if qualified_funds is not None and not qualified_funds.empty:
            print(qualified_funds[['基金代码', '基金简称', '日增长率']])
            
            qualified_funds.to_csv('qualified_funds.csv', index=False, encoding='utf_8_sig')
            print("\n结果已保存至 qualified_funds.csv 文件。")
        else:
            print("没有找到符合条件的基金。请尝试调整筛选条件。")
