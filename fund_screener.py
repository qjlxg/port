import pandas as pd
import warnings

# 忽略 pandas 在解析 HTML 时可能出现的警告
warnings.filterwarnings('ignore', category=UserWarning)

# --- 步骤 1: 自动下载基金数据 ---
def download_fund_data(url='http://fund.eastmoney.com/fund.html'):
    """
    从天天基金网下载基金数据，并返回一个 Pandas DataFrame。
    """
    print("正在尝试从天天基金网下载数据...")
    try:
        # 使用 pandas 的 read_html 函数，并指定解析器
        tables = pd.read_html(url, encoding='gbk')

        # 找到正确的基金数据表格
        df = None
        for table in tables:
            # 根据列名判断是否为目标表格
            if '基金代码' in table.columns and '基金简称' in table.columns:
                df = table
                break
        
        if df is None:
            print("未找到基金数据表格。请检查网站结构是否变化。")
            return None
        
        print("数据下载成功，正在进行处理...")

        # 清理多层表头
        df.columns = df.columns.droplevel(0)
        df.columns = df.iloc[0]
        df = df[1:].copy()

        # 建立一个列名映射字典，以应对网站列名变化
        col_mapping = {
            '基金代码': '基金代码',
            '基金简称': '基金简称',
            '近1年': '近1年收益率',
            '近3年': '近3年收益率',
            '最大回撤(%)': '最大回撤',  # 这是可能的列名，如有不同请修改
            '近6月': '近6月收益率',
            '近2年': '近2年收益率',
            '成立来': '成立来收益率',
            '单位净值': '单位净值'
        }
        
        # 使用映射字典重命名列
        df.rename(columns=col_mapping, inplace=True)
        
        # 打印所有找到的列名，以供调试
        print("以下是下载数据中找到的列名：")
        print(df.columns.tolist())
        
        # 将需要的列转换为数值类型
        required_cols = ['近1年收益率', '近3年收益率', '最大回撤']
        for col in required_cols:
            if col in df.columns:
                # 移除百分号并处理空值，转换为数值
                df[col] = pd.to_numeric(df[col].astype(str).str.rstrip('%').replace('--', '0'), errors='coerce') / 100
            else:
                print(f"警告：找不到列 '{col}'。请检查网站列名是否变化。")
                return None
                
        return df

    except Exception as e:
        print(f"发生错误：{e}")
        print("请确保已安装 pandas, lxml, html5lib 和 beautifulsoup4。")
        print("可以使用命令: pip install pandas lxml html5lib beautifulsoup4")
        return None

# --- 步骤 2: 设定筛选条件并执行筛选 ---
def screen_funds(df):
    """
    根据预设条件筛选基金。
    """
    if df is None:
        return None

    print("\n开始执行基金筛选...")

    # 定义你的筛选条件
    min_1y_return = 0.10   # 近1年收益率 > 10%
    min_3y_return = 0.20   # 近3年收益率 > 20%
    max_drawdown = 0.15    # 最大回撤 < 15%

    filtered_funds = df[
        (df['基金类型'].isin(['混合型', '股票型'])) &
        (df['近1年收益率'] > min_1y_return) &
        (df['近3年收益率'] > min_3y_return) &
        (df['最大回撤'] < max_drawdown)
    ].copy()

    return filtered_funds

# --- 步骤 3: 运行主程序 ---
if __name__ == "__main__":
    fund_data_df = download_fund_data()
    
    if fund_data_df is not None:
        qualified_funds = screen_funds(fund_data_df)
        
        print("\n--- 符合筛选条件的基金列表 ---")
        if not qualified_funds.empty:
            # 打印筛选结果，只展示关键信息
            print(qualified_funds[['基金代码', '基金简称', '近1年收益率', '近3年收益率', '最大回撤']])
            
            # 也可以选择将筛选结果保存到新的CSV文件
            qualified_funds[['基金代码', '基金简称', '近1年收益率', '近3年收益率', '最大回撤']].to_csv('qualified_funds.csv', index=False, encoding='utf_8_sig')
            print("\n结果已保存至 qualified_funds.csv 文件。")
        else:
            print("没有找到符合条件的基金。请尝试调整筛选条件。")
