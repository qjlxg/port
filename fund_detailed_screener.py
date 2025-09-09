import pandas as pd
import aiohttp
import asyncio
import re
from typing import List, Dict, Any, Tuple, Optional
import random
from bs4 import BeautifulSoup

# 随机 User-Agent 列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'
]

async def fetch_web_data_async(session: aiohttp.ClientSession, url: str) -> Tuple[Optional[str], Optional[str]]:
    """通用异步网页数据抓取函数"""
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    try:
        async with session.get(url, headers=headers, timeout=30) as response:
            response.raise_for_status()
            return await response.text(), None
    except aiohttp.ClientError as e:
        return None, f"请求失败: {e}"
    except asyncio.TimeoutError:
        return None, "请求超时"

async def get_manager_info(session: aiohttp.ClientSession, code: str) -> Tuple[Optional[str], Optional[float], Optional[int], Optional[str]]:
    """异步获取基金经理信息"""
    url = f"https://fund.eastmoney.com/{code}.html"
    html, error = await fetch_web_data_async(session, url)
    if error:
        return None, None, None, error
    
    try:
        manager_match = re.search(r'基金经理：<a.*?>(.*?)</a>', html, re.DOTALL)
        manager_name = manager_match.group(1).strip() if manager_match else 'N/A'

        tenure_match = re.search(r'从业年限：<span>(.*?)年', html)
        tenure_years = float(tenure_match.group(1)) if tenure_match else 0.0

        fund_count_match = re.search(r'现任基金数：<span>(.*?)只', html)
        fund_count = int(fund_count_match.group(1)) if fund_count_match else 0
        
        return manager_name, tenure_years, fund_count, None
    except Exception as e:
        return None, None, None, f"解析基金经理信息失败: {e}"

async def get_holdings_info(session: aiohttp.ClientSession, code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """异步获取前十大持仓信息"""
    url = f"https://fundf10.eastmoney.com/ccmx_{code}.html"
    html, error = await fetch_web_data_async(session, url)
    if error:
        return None, None, error

    try:
        soup = BeautifulSoup(html, 'lxml')
        top_10_stocks = []
        
        table = soup.find('table', class_='w782')
        if table:
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    stock_name = cols[1].get_text(strip=True)
                    proportion = cols[3].get_text(strip=True)
                    top_10_stocks.append(f"{stock_name}({proportion})")
        
        holdings_str = " | ".join(top_10_stocks)
        if not holdings_str:
            holdings_str = "无持仓数据"
        
        date_span = soup.find('span', string=re.compile(r'截止至：|截止日期：'))
        update_date = date_span.next_sibling.strip() if date_span and date_span.next_sibling else "N/A"
        
        return holdings_str, update_date, None
    except Exception as e:
        return None, None, f"解析持仓信息失败: {e}"

async def process_fund_details(fund: Dict[str, Any], session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> Optional[Dict[str, Any]]:
    """处理单个基金的详细信息抓取"""
    async with semaphore:
        code_str = str(fund['基金代码']).zfill(6)
        print(f"正在获取基金 {code_str} 的详细信息...", flush=True)

        await asyncio.sleep(random.uniform(1, 3))

        manager_name, tenure_years, fund_count, manager_error = await get_manager_info(session, code_str)
        if manager_error:
            print(f"基金 {code_str} 基金经理信息获取失败: {manager_error}", flush=True)

        holdings_str, update_date, holdings_error = await get_holdings_info(session, code_str)
        if holdings_error:
            print(f"基金 {code_str} 持仓信息获取失败: {holdings_error}", flush=True)

        fund['基金经理'] = manager_name
        fund['从业年限 (年)'] = tenure_years
        fund['现任基金数 (只)'] = fund_count
        fund['前十大持仓'] = holdings_str
        fund['持仓更新日期'] = update_date
        
        return fund

async def main():
    try:
        df_funds = pd.read_csv('recommended_cn_funds.csv', dtype={'基金代码': str})
    except FileNotFoundError:
        print("错误：未找到文件 recommended_cn_funds.csv。请先运行 fund_screener.py", flush=True)
        return

    print(f"已加载 {len(df_funds)} 只基金，开始获取详细信息。", flush=True)

    semaphore = asyncio.Semaphore(5)
    
    conn = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [process_fund_details(fund.to_dict(), session, semaphore) for _, fund in df_funds.iterrows()]
        
        enriched_funds = await asyncio.gather(*tasks, return_exceptions=True)
        
    final_results = [item for item in enriched_funds if not isinstance(item, Exception)]
    
    if final_results:
        final_df = pd.DataFrame(final_results)
        final_df.to_csv('fund_details_enriched.csv', index=False, encoding='utf-8-sig')
        print(f"\n成功获取并保存 {len(final_df)} 只基金的详细信息至 fund_details_enriched.csv", flush=True)
        print("\n部分详细信息预览：", flush=True)
        print(final_df[['基金代码', '基金名称', '基金经理', '前十大持仓', '综合评分']].head(), flush=True)
    else:
        print("\n抱歉，未能获取任何基金的详细信息。请检查网络或稍后重试。", flush=True)

if __name__ == '__main__':
    asyncio.run(main())
