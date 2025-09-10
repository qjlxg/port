import os
import re
import signal
from typing import List, Union, Dict

import threading
from bs4 import BeautifulSoup
import multitasking
import pandas as pd
import requests
import rich
from jsonpath import jsonpath
from retry import retry
from tqdm.auto import tqdm
import time
import numpy as np
import json
from datetime import datetime, timedelta
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor
import pickle
import warnings
import traceback

# 忽略警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("module")


# 筛选条件
MIN_RETURN = 3.0  # 年化收益率 ≥ 3%
MAX_VOLATILITY = 25.0  # 波动率 ≤ 25%
MIN_SHARPE = 0.2  # 夏普比率 ≥ 0.2
MAX_FEE = 2.5  # 管理费 ≤ 2.5%
RISK_FREE_RATE = 3.0  # 无风险利率 3%
MIN_DAYS = 100  # 最低数据天数
TIMEOUT = 10  # 网络请求超时时间（秒）
FUND_TYPE_FILTER = ['混合型', '股票型', '指数型']  # 基金类型筛选

# 配置 requests 重试机制
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))
fund_session = session

# 随机 User-Agent 和 Headers
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

# 扩展的申万行业分类数据
SW_INDUSTRY_MAPPING = {
    '600519': '食品饮料', '000858': '食品饮料', '002475': '家用电器', '002415': '家用电器',
    '300750': '计算机', '300059': '传媒', '002460': '汽车', '600036': '金融',
    '600276': '医药生物', '600030': '金融',
    '000001': '金融', '600000': '金融', '601318': '金融', '601166': '金融',
    '000333': '家用电器', '000651': '家用电器', '600690': '家用电器',
    '002304': '食品饮料', '000568': '食品饮料', '600809': '食品饮料', '603288': '食品饮料',
    '300760': '医药生物', '002714': '农林牧渔', '601012': '电力设备', '300274': '电力设备',
    '601688': '金融', '600837': '金融', '601398': '金融', '601288': '金融',
    '002241': '计算机', '300033': '计算机', '002594': '汽车', '601633': '汽车',
    '603259': '医药生物', '300122': '医药生物', '600196': '医药生物',
    '000423': '医药生物', '002007': '医药生物', '600085': '医药生物',
    '600660': '汽车', '002920': '计算机', '300628': '计算机',
    '600893': '电力设备', '300014': '电力设备', '601985': '电力设备',
    '002027': '传媒', '300027': '传媒', '002739': '传媒',
    '000725': '电子', '300223': '电子', '600584': '电子',
    '600887': '食品饮料', '603888': '食品饮料'
}

# 数据缓存目录
CACHE_DIR = "fund_data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


class EastmoneyFundHeaders(object):
    """东方财富基金数据请求头"""
    UserAgent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36 MicroMessenger/6.5.2.501 NetType/WIFI WindowsWechat'
    Referer = 'https://servicewechat.com/wxf138096181f69742/10/page-frame.html'
    Cookie = 'qgqp_b_id=a8600d81c3c3a4f2105e46648d79555c'


class MagicConfig(object):
    """
    魔法变量配置类
    """
    RETURN_DF = '__RETURN_DF__'


def to_numeric(func):
    """
    装饰器: 将 DataFrame 中所有列转换为数值类型
    """

    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        if df.empty:
            return df
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    return wrapper


if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGINT, multitasking.killall)

MAX_CONNECTIONS = 100


def get_all_funds_from_eastmoney():
    cache_file = os.path.join(CACHE_DIR, "fund_list.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                funds_df = pickle.load(f)
            print(f"    √ 从缓存加载 {len(funds_df)} 只基金。", flush=True)
            return funds_df
        except Exception as e:
            print(f"    × 加载基金列表缓存失败: {e}，将重新获取。", flush=True)

    print(">>> 步骤1: 正在动态获取全市场基金列表...", flush=True)
    url = "http://fund.eastmoney.com/js/fundcode_search.js"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': 'http://fund.eastmoney.com/',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Connection': 'keep-alive'
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        content = response.text
        match = re.search(r'var\s+r\s*=\s*(\[.*?\]);', content, re.DOTALL)
        if match:
            fund_data = json.loads(match.group(1))
            df = pd.DataFrame(fund_data, columns=['code', 'pinyin', 'name', 'type', 'pinyin_full'])
            df = df[['code', 'name', 'type']].drop_duplicates(subset=['code'])
            df = df[df['type'].isin(FUND_TYPE_FILTER)].copy()
            print(f"    √ 获取到 {len(df)} 只{', '.join(FUND_TYPE_FILTER)}基金。", flush=True)
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
            return df
        print("    × 未能解析基金列表数据。", flush=True)
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"    × 获取基金列表失败: {e}", flush=True)
        return pd.DataFrame()
    except Exception as e:
        print(f"    × 解析基金列表时发生异常: {e}", flush=True)
        return pd.DataFrame()


@retry(tries=3)
@to_numeric
def get_quote_history(fund_code: str, pz: int = 40000) -> pd.DataFrame:
    """
    根据基金代码和要获取的页码抓取基金净值信息
    """
    data = {
        "FCODE": f"{fund_code}",
        "IsShareNet": "true",
        "MobileKey": "1",
        "appType": "ttjj",
        "appVersion": "6.2.8",
        "cToken": "1",
        "deviceid": "1",
        "pageIndex": "1",
        "pageSize": f"{pz}",
        "plat": "Iphone",
        "product": "EFund",
        "serverVersion": "6.2.8",
        "uToken": "1",
        "userId": "1",
        "version": "6.2.8",
    }
    url = "https://fundmobapi.eastmoney.com/FundMNewApi/FundMNHisNetList"
    json_response = fund_session.get(
        url, headers=EastmoneyFundHeaders, data=data, verify=False
    ).json()
    rows = []
    columns = ["日期", "单位净值", "累计净值", "涨跌幅"]
    if json_response is None:
        return pd.DataFrame(rows, columns=columns)
    datas = json_response["Datas"]
    if len(datas) == 0:
        return pd.DataFrame(rows, columns=columns)
    rows = []
    for stock in datas:
        date = stock["FSRQ"]
        rows.append(
            {
                "日期": date,
                "单位净值": stock["DWJZ"],
                "累计净值": stock["LJJZ"],
                "涨跌幅": stock["JZZZL"],
            }
        )
    df = pd.DataFrame(rows)
    return df


def get_quote_history_multi(
    fund_codes: List[str], pz: int = 40000, **kwargs
) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    pbar = tqdm(total=len(fund_codes))

    @multitasking.task
    @retry(tries=3, delay=1)
    def start(fund_code: str):
        if len(multitasking.get_active_tasks()) >= MAX_CONNECTIONS:
            time.sleep(3)
        _df = get_quote_history(fund_code, pz)
        dfs[fund_code] = _df
        pbar.update(1)
        pbar.set_description_str(f"Processing => {fund_code}")

    for f in fund_codes:
        start(f)
    multitasking.wait_for_tasks()
    pbar.close()
    if kwargs.get(MagicConfig.RETURN_DF):
        return pd.concat(dfs, axis=0, ignore_index=True)
    return dfs


@retry(tries=3)
@to_numeric
def get_realtime_increase_rate(fund_codes: Union[List[str], str]) -> pd.DataFrame:
    """
    获取基金实时估算涨跌幅度
    """
    if not isinstance(fund_codes, list):
        fund_codes = [fund_codes]
    data = {
        "pageIndex": "1",
        "pageSize": "300000",
        "Sort": "",
        "Fcodes": ",".join(fund_codes),
        "SortColumn": "",
        "IsShowSE": "false",
        "P": "F",
        "deviceid": "3EA024C2-7F22-408B-95E4-383D38160FB3",
        "plat": "Iphone",
        "product": "EFund",
        "version": "6.2.8",
    }
    columns = {
        "FCODE": "基金代码",
        "SHORTNAME": "基金名称",
        "ACCNAV": "最新净值",
        "PDATE": "最新净值公开日期",
        "GZTIME": "估算时间",
        "GSZZL": "估算涨跌幅",
    }
    url = "https://fundmobapi.eastmoney.com/FundMNewApi/FundMNFInfo"
    json_response = fund_session.get(
        url, headers=EastmoneyFundHeaders, data=data
    ).json()
    rows = jsonpath(json_response, "$..Datas[:]")
    if not rows:
        df = pd.DataFrame(columns=columns.values())
        return df
    df = pd.DataFrame(rows).rename(columns=columns)[columns.values()]
    return df


@retry(tries=3)
def get_fund_codes(ft: str = None) -> pd.DataFrame:
    """
    获取天天基金网公开的全部公墓基金名单
    """
    params = [
        ("op", "dy"),
        ("dt", "kf"),
        ("rs", ""),
        ("gs", "0"),
        ("sc", "qjzf"),
        ("st", "desc"),
        ("es", "0"),
        ("qdii", ""),
        ("pi", "1"),
        ("pn", "50000"),
        ("dx", "0"),
    ]
    headers = {
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36 Edg/87.0.664.75",
        "Accept": "*/*",
        "Referer": "http://fund.eastmoney.com/data/fundranking.html",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    }
    if ft is not None:
        params.append(("ft", ft))
    url = "http://fund.eastmoney.com/data/rankhandler.aspx"
    response = fund_session.get(url, headers=headers, params=params)
    columns = ["基金代码", "基金简称"]
    results = re.findall('"(\d{6}),(.*?),', response.text)
    df = pd.DataFrame(results, columns=columns)
    return df


@retry(tries=3)
def get_fund_manager(ft: str) -> pd.DataFrame:
    url = f"http://fundf10.eastmoney.com/jjjl_{ft}.html"
    response = fund_session.get(url)
    if not response:
        return pd.DataFrame()
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    contents = soup.find("div", class_="bs_gl").find_all("label")
    start_date = contents[0].span.text
    managers = ";".join([a.text for a in contents[1].find_all("a")])
    type_str = contents[2].span.text
    company = contents[3].find("a").text
    share = contents[4].span.text.replace("\r", "").replace("\n", "").replace(" ", "")
    return pd.DataFrame(
        data=[
            [
                ft,
                start_date,
                company,
                managers,
                type_str,
                share,
                str(pd.to_datetime("today").date()),
            ]
        ],
        columns=[
            "基金代码",
            "基金经理任职日期",
            "基金公司",
            "基金经理",
            "基金种类",
            "基金规模",
            "当前日期",
        ],
    )


@retry(tries=3)
@to_numeric
def get_invest_position(
    fund_code: str, dates: Union[str, List[str]] = None
) -> pd.DataFrame:
    """
    获取基金持仓占比数据
    """
    columns = {
        "GPDM": "股票代码",
        "GPJC": "股票简称",
        "JZBL": "持仓占比",
        "PCTNVCHG": "较上期变化",
    }
    df = pd.DataFrame(columns=columns.values())
    if not isinstance(dates, List):
        dates = [dates]
    if dates is None:
        dates = [None]
    dfs: List[pd.DataFrame] = []
    for date in dates:
        params = [
            ("FCODE", fund_code),
            ("appType", "ttjj"),
            ("deviceid", "3EA024C2-7F22-408B-95E4-383D38160FB3"),
            ("plat", "Iphone"),
            ("product", "EFund"),
            ("serverVersion", "6.2.8"),
            ("version", "6.2.8"),
        ]
        if date is not None:
            params.append(("DATE", date))
        url = "https://fundmobapi.eastmoney.com/FundMNewApi/FundMNInverstPosition"
        json_response = fund_session.get(
            url, headers=EastmoneyFundHeaders, params=params
        ).json()
        stocks = jsonpath(json_response, "$..fundStocks[:]")
        if not stocks:
            continue
        date = json_response["Expansion"]
        _df = pd.DataFrame(stocks)
        _df["公开日期"] = date
        _df.insert(0, "基金代码", fund_code)
        dfs.append(_df)
    fields = ["基金代码"] + list(columns.values()) + ["公开日期"]
    if not dfs:
        return pd.DataFrame(columns=fields)
    df = pd.concat(dfs, axis=0, ignore_index=True).rename(columns=columns)[fields]
    return df


@retry(tries=3)
@to_numeric
def get_period_change(fund_code: str) -> pd.DataFrame:
    """
    获取基金阶段涨跌幅度
    """
    params = (
        ("AppVersion", "6.3.8"),
        ("FCODE", fund_code),
        ("MobileKey", "3EA024C2-7F22-408B-95E4-383D38160FB3"),
        ("OSVersion", "14.3"),
        ("deviceid", "3EA024C2-7F22-408B-95E4-383D38160FB3"),
        ("passportid", "3061335960830820"),
        ("plat", "Iphone"),
        ("product", "EFund"),
        ("version", "6.3.6"),
    )
    url = "https://fundmobapi.eastmoney.com/FundMNewApi/FundMNPeriodIncrease"
    json_response = fund_session.get(
        url, headers=EastmoneyFundHeaders, params=params
    ).json()
    columns = {
        "syl": "收益率",
        "avg": "同类平均",
        "rank": "同类排行",
        "sc": "同类总数",
        "title": "时间段",
    }
    titles = {
        "Z": "近一周",
        "Y": "近一月",
        "3Y": "近三月",
        "6Y": "近六月",
        "1N": "近一年",
        "2Y": "近两年",
        "3N": "近三年",
        "5N": "近五年",
        "JN": "今年以来",
        "LN": "成立以来",
    }
    # 发行时间
    ESTABDATE = json_response["Expansion"]["ESTABDATE"]
    df = pd.DataFrame(json_response["Datas"])
    df = df[list(columns.keys())].rename(columns=columns)
    df["时间段"] = titles.values()
    df.insert(0, "基金代码", fund_code)
    return df


def get_public_dates(fund_code: str) -> List[str]:
    """
    获取历史上更新持仓情况的日期列表
    """
    params = (
        ("FCODE", fund_code),
        ("appVersion", "6.3.8"),
        ("deviceid", "3EA024C2-7F22-408B-95E4-383D38160FB3"),
        ("plat", "Iphone"),
        ("product", "EFund"),
        ("serverVersion", "6.3.6"),
        ("version", "6.3.8"),
    )
    url = "https://fundmobapi.eastmoney.com/FundMNewApi/FundMNIVInfoMultiple"
    json_response = fund_session.get(
        url, headers=EastmoneyFundHeaders, params=params
    ).json()
    if json_response["Datas"] is None:
        return []
    return json_response["Datas"]


@retry(tries=3)
@to_numeric
def get_types_percentage(
    fund_code: str, dates: Union[List[str], str, None] = None
) -> pd.DataFrame:
    """
    获取指定基金不同类型占比信息
    """
    columns = {
        "GP": "股票比重",
        "ZQ": "债券比重",
        "HB": "现金比重",
        "JZC": "总规模(亿元)",
        "QT": "其他比重",
    }
    df = pd.DataFrame(columns=columns.values())
    if not isinstance(dates, List):
        dates = [dates]
    elif dates is None:
        dates = [None]
    for date in dates:
        params = [
            ("FCODE", fund_code),
            ("OSVersion", "14.3"),
            ("appVersion", "6.3.8"),
            ("deviceid", "3EA024C2-7F21-408B-95E4-383D38160FB3"),
            ("plat", "Iphone"),
            ("product", "EFund"),
            ("serverVersion", "6.3.6"),
            ("version", "6.3.8"),
        ]
        if date is not None:
            params.append(("DATE", date))
        params = tuple(params)
        url = "https://fundmobapi.eastmoney.com/FundMNewApi/FundMNAssetAllocationNew"
        json_response = fund_session.get(
            url, params=params, headers=EastmoneyFundHeaders
        ).json()
        if len(json_response["Datas"]) == 0:
            continue
        _df = pd.DataFrame(json_response["Datas"])[columns.keys()]
        _df = _df.rename(columns=columns)
        df = pd.concat([df, _df], axis=0, ignore_index=True)
    df.insert(0, "基金代码", fund_code)
    return df


@retry(tries=3)
@to_numeric
def get_base_info_single(fund_code: str) -> pd.Series:
    """
    获取基金的一些基本信息
    """
    params = (
        ("FCODE", fund_code),
        ("deviceid", "3EA024C2-7F22-408B-95E4-383D38160FB3"),
        ("plat", "Iphone"),
        ("product", "EFund"),
        ("version", "6.3.8"),
    )
    url = "https://fundmobapi.eastmoney.com/FundMNewApi/FundMNNBasicInformation"
    json_response = fund_session.get(
        url, headers=EastmoneyFundHeaders, params=params
    ).json()
    columns = {
        "FCODE": "基金代码",
        "SHORTNAME": "基金简称",
        "ESTABDATE": "成立日期",
        "RZDF": "涨跌幅",
        "DWJZ": "最新净值",
        "JJGS": "基金公司",
        "FSRQ": "净值更新日期",
        "COMMENTS": "简介",
    }
    items = json_response["Datas"]
    if not items:
        rich.print("基金代码", fund_code, "可能有误")
        return pd.Series(index=columns.values())
    s = pd.Series(json_response["Datas"]).rename(index=columns)[columns.values()]
    s = s.apply(lambda x: x.replace("\n", " ").strip() if isinstance(x, str) else x)
    return s


def get_base_info_muliti(fund_codes: List[str]) -> pd.Series:
    """
    获取多只基金基本信息
    """
    ss = []
    @multitasking.task
    @retry(tries=3, delay=1)
    def start(fund_code: str) -> None:
        s = get_base_info_single(fund_code)
        ss.append(s)
        pbar.update()
        pbar.set_description(f"Processing => {fund_code}")
    pbar = tqdm(total=len(fund_codes))
    for fund_code in fund_codes:
        start(fund_code)
    multitasking.wait_for_tasks()
    df = pd.DataFrame(ss)
    return df


def get_base_info(fund_codes: Union[str, List[str]]) -> Union[pd.Series, pd.DataFrame]:
    """
    获取基金的一些基本信息
    """
    if isinstance(fund_codes, str):
        return get_base_info_single(fund_codes)
    elif hasattr(fund_codes, "__iter__"):
        return get_base_info_muliti(fund_codes)
    raise TypeError(f"所给的 {fund_codes} 不符合参数要求")


@to_numeric
def get_industry_distribution(
    fund_code: str, dates: Union[str, List[str]] = None
) -> pd.DataFrame:
    """
    获取指定基金行业分布信息
    """
    columns = {
        "HYMC": "行业名称",
        "ZJZBL": "持仓比例",
        "FSRQ": "公布日期",
        "SZ": "市值",
    }
    df = pd.DataFrame(columns=columns.values())
    if isinstance(dates, str):
        dates = [dates]
    elif dates is None:
        dates = [None]
    for date in dates:
        params = [
            ("FCODE", fund_code),
            ("OSVersion", "14.4"),
            ("appVersion", "6.3.8"),
            ("deviceid", "3EA024C2-7F22-408B-95E4-383D38160FB3"),
            ("plat", "Iphone"),
            ("product", "EFund"),
            ("serverVersion", "6.3.6"),
            ("version", "6.3.8"),
        ]
        if date is not None:
            params.append(("DATE", date))
        url = "https://fundmobapi.eastmoney.com/FundMNewApi/FundMNSectorAllocation"
        response = fund_session.get(url, headers=EastmoneyFundHeaders, params=params)
        datas = response.json()["Datas"]
        _df = pd.DataFrame(datas)
        _df = _df.rename(columns=columns)
        df = pd.concat([df, _df], axis=0, ignore_index=True)
    df.insert(0, "基金代码", fund_code)
    df = df.drop_duplicates()
    return df


def get_pdf_reports(fund_code: str, max_count: int = 12, save_dir: str = "pdf") -> None:
    """
    根据基金代码获取其全部 pdf 报告
    """

    headers = {
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.128 Safari/537.36 Edg/89.0.774.77",
        "Accept": "*/*",
        "Referer": "http://fundf10.eastmoney.com/",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    }
    @multitasking.task
    @retry(tries=3, delay=1)
    def download_file(
        fund_code: str, url: str, filename: str, file_type=".pdf"
    ) -> None:
        """
        根据文件名、文件直链等参数下载文件
        """
        pbar.set_description(f"Processing => {fund_code}")
        fund_code = str(fund_code)
        if not os.path.exists(save_dir + "/" + fund_code):
            os.mkdir(save_dir + "/" + fund_code)
        response = fund_session.get(url, headers=headers)
        path = f"{save_dir}/{fund_code}/{filename}{file_type}"
        with open(path, "wb") as f:
            f.write(response.content)
        if os.path.getsize(path) == 0:
            os.remove(path)
            return
        pbar.update(1)
    params = (
        ("fundcode", fund_code),
        ("pageIndex", "1"),
        ("pageSize", "200000"),
        ("type", "3"),
    )
    json_response = fund_session.get(
        "http://api.fund.eastmoney.com/f10/JJGG", headers=headers, params=params
    ).json()
    base_link = "http://pdf.dfcfw.com/pdf/H2_{}_1.pdf"
    pbar = tqdm(total=min(max_count, len(json_response["Data"])))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for item in json_response["Data"][-max_count:]:
        title = item["TITLE"]
        download_url = base_link.format(item["ID"])
        download_file(fund_code, download_url, title)
    multitasking.wait_for_tasks()
    pbar.close()
    print(f"{fund_code} 的 pdf 文件已存储到文件夹 {save_dir}/{fund_code} 中")


def get_fund_net_values(code, start_date, end_date):
    cache_file = os.path.join(CACHE_DIR, f"net_values_{code}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                net_df, latest_value = pickle.load(f)
            if not net_df.empty and len(net_df) >= MIN_DAYS:
                return net_df, latest_value, 'cache'
        except Exception:
            pass
    df, latest_value = get_net_values_from_pingzhongdata(code, start_date, end_date)
    if not df.empty and len(df) >= MIN_DAYS:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((df, latest_value), f)
        except Exception:
            pass
        return df, latest_value, 'pingzhongdata'
    df, latest_value = get_net_values_from_lsjz(code, start_date, end_date)
    if not df.empty and len(df) >= MIN_DAYS:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((df, latest_value), f)
        except Exception:
            pass
        return df, latest_value, 'lsjz'
    return pd.DataFrame(), None, 'None'


def get_net_values_from_pingzhongdata(code, start_date, end_date):
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Accept': 'text/javascript, application/javascript, */*',
        'Connection': 'keep-alive'
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        net_worth_match = re.search(r'Data_netWorthTrend\s*=\s*(\[.*?\]);', response.text, re.DOTALL)
        if not net_worth_match:
            print(f"    调试: {url} 未找到净值数据。", flush=True)
            return pd.DataFrame(), None
        net_worth_list = json.loads(net_worth_match.group(1))
        df = pd.DataFrame(net_worth_list).rename(columns={'x': 'date', 'y': 'net_value'})
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
        df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)
        latest_value = df['net_value'].iloc[-1] if not df.empty else None
        return df, latest_value
    except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError) as e:
        print(f"    调试: {url} 接口请求或JSON解析失败: {e}", flush=True)
        return pd.DataFrame(), None


def get_net_values_from_lsjz(code, start_date, end_date):
    url = f"http://fund.eastmoney.com/f10/lsjz?fundCode={code}&pageIndex=1&pageSize=50000"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/f10/fjcc_{code}.html',
        'Accept': 'application/json, text/plain, */*',
        'Connection': 'keep-alive'
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        data_str_match = re.search(r'var\s+apidata=\{content:"(.*?)",', response.text, re.DOTALL)
        if not data_str_match:
            print(f"    调试: {url} 未找到历史净值数据。", flush=True)
            return pd.DataFrame(), None
        json_data_str = data_str_match.group(1).replace("\\", "")
        data = json.loads(json_data_str)
        if 'LSJZList' in data and data['LSJZList']:
            df = pd.DataFrame(data['LSJZList']).rename(columns={'FSRQ': 'date', 'DWJZ': 'net_value'})
            df['date'] = pd.to_datetime(df['date'])
            df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
            df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
            df = df.sort_values('date').dropna(subset=['net_value']).reset_index(drop=True)
            latest_value = df['net_value'].iloc[-1] if not df.empty else None
            return df, latest_value
        return pd.DataFrame(), None
    except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError) as e:
        print(f"    调试: {url} 接口请求或JSON解析失败: {e}", flush=True)
        return pd.DataFrame(), None


def get_fund_realtime_estimate(code):
    cache_file = os.path.join(CACHE_DIR, f"realtime_estimate_{code}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    url = f"http://fundgz.1234567.com.cn/js/{code}.js?rt={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Accept': 'application/json, text/javascript, */*',
        'Connection': 'keep-alive'
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        match = re.search(r'jsonpgz\((.*)\)', response.text, re.DOTALL)
        if match:
            json_data = json.loads(match.group(1))
            gsz = json_data.get('gsz')
            if gsz:
                try:
                    gsz_float = float(gsz)
                    with open(cache_file, "wb") as f:
                        pickle.dump(gsz_float, f)
                    return gsz_float
                except (ValueError, TypeError):
                    pass
            return None
    except Exception as e:
        print(f"    调试: 获取实时估值 {code} 异常: {e}", flush=True)
        return None


def get_fund_fee(code):
    cache_file = os.path.join(CACHE_DIR, f"fee_{code}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js?v={int(time.time() * 1000)}"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/{code}.html',
        'Accept': 'text/javascript, application/javascript, */*',
        'Connection': 'keep-alive'
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        fee_match = re.search(r'data_fundTribble\.ManagerFee=\'([\d.]+)\'', response.text)
        fee = float(fee_match.group(1)) if fee_match else 1.5
        with open(cache_file, "wb") as f:
            pickle.dump(fee, f)
        return fee
    except requests.exceptions.RequestException:
        print(f"    调试: 获取管理费 {code} 请求失败。", flush=True)
        return 1.5
    except Exception as e:
        print(f"    调试: 获取管理费 {code} 异常: {e}", flush=True)
        return 1.5


def get_fund_company_info(code):
    cache_file = os.path.join(CACHE_DIR, f"company_{code}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    url = f"http://fund.eastmoney.com/f10/jbgk_{code}.html"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': f'http://fund.eastmoney.com/f10/tsdata_{code}.html',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Connection': 'keep-alive'
    }
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        company_name = soup.find('div', class_='fundDetail-main').find('a').text
        with open(cache_file, "wb") as f:
            pickle.dump(company_name, f)
        return company_name
    except Exception as e:
        print(f"    调试: 获取基金公司 {code} 异常: {e}", flush=True)
        return None


def get_industry_from_stocks(stocks):
    if not stocks:
        return pd.DataFrame(), 0.0
    df = pd.DataFrame(stocks, columns=['code', 'proportion'])
    df['industry'] = df['code'].map(SW_INDUSTRY_MAPPING).fillna('其他')
    industry_df = df.groupby('industry')['proportion'].sum().reset_index()
    industry_df.columns = ['行业名称', '持仓比例']
    industry_df = industry_df.sort_values('持仓比例', ascending=False).reset_index(drop=True)
    top3_concentration = industry_df['持仓比例'].iloc[:3].sum() if len(industry_df) >= 3 else 0.0
    return industry_df, top3_concentration


def calculate_metrics(df):
    if df.empty or len(df) < MIN_DAYS:
        return None
    df['daily_return'] = df['net_value'].pct_change()
    if df['daily_return'].isnull().all():
        return None
    # 年化收益率
    annual_return = (df['net_value'].iloc[-1] / df['net_value'].iloc[0]) ** (252 / len(df)) - 1
    # 年化波动率
    annual_volatility = df['daily_return'].std() * np.sqrt(252)
    # 夏普比率
    sharpe_ratio = (annual_return - RISK_FREE_RATE / 100) / annual_volatility if annual_volatility > 0 else 0
    return {
        '年化收益率 (%)': annual_return * 100,
        '年化波动率 (%)': annual_volatility * 100,
        '夏普比率': sharpe_ratio,
    }


def calculate_score_and_filter(fund, start_date):
    code, name, fund_type = fund['code'], fund['name'], fund['type']
    print(f"\n正在处理 {name}({code})...", flush=True)
    net_df, latest_value, source = get_fund_net_values(code, start_date, datetime.now())
    if net_df.empty:
        print(f"    × {name} 净值数据不足，跳过。({source})", flush=True)
        return None
    metrics = calculate_metrics(net_df)
    if not metrics:
        print(f"    × {name} 指标计算失败，数据不足{MIN_DAYS}天，跳过。", flush=True)
        return None
    fee = get_fund_fee(code)
    realtime_estimate = get_fund_realtime_estimate(code)
    company = get_fund_company_info(code)
    # 持仓信息
    industry_df = pd.DataFrame()
    top3_concentration = 0.0
    try:
        url_stock = f"http://fund.eastmoney.com/f10/FundHoldInfo.aspx?fundcode={code}&ccdm=lc"
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        response = session.get(url_stock, headers=headers, timeout=TIMEOUT)
        soup = BeautifulSoup(response.text, 'html.parser')
        stock_tables = soup.find_all('table', class_='tstj')
        if stock_tables:
            stock_rows = stock_tables[0].find_all('tr')[1:]
            stocks_data = []
            for tr in stock_rows:
                tds = tr.find_all('td')
                if len(tds) >= 4:
                    stock_code = tds[1].text
                    proportion = float(tds[3].text.replace('%', ''))
                    stocks_data.append({'code': stock_code, 'proportion': proportion})
            industry_df, top3_concentration = get_industry_from_stocks(stocks_data)
    except Exception as e:
        print(f"    调试: 获取持仓数据 {code} 异常: {e}", flush=True)
    # 评分逻辑
    score = 0
    reason_list = []
    # 1. 年化收益率
    if metrics['年化收益率 (%)'] >= MIN_RETURN:
        score += 30
    else:
        reason_list.append(f"收益率({metrics['年化收益率 (%)']:.2f}%)过低")
    # 2. 波动率
    if metrics['年化波动率 (%)'] <= MAX_VOLATILITY:
        score += 25
    else:
        reason_list.append(f"波动率({metrics['年化波动率 (%)']:.2f}%)过高")
    # 3. 夏普比率
    if metrics['夏普比率'] >= MIN_SHARPE:
        score += 20
    else:
        reason_list.append(f"夏普比率({metrics['夏普比率']:.2f})过低")
    # 4. 管理费
    if fee <= MAX_FEE:
        score += 15
    else:
        reason_list.append(f"管理费({fee:.2f}%)过高")
    # 5. 行业集中度
    if top3_concentration <= 60:
        score += 10
    else:
        reason_list.append(f"行业集中度({top3_concentration:.2f}%)过高")
    if not reason_list:
        reason = "所有条件均满足"
    else:
        reason = "未通过: " + ", ".join(reason_list)
    result = {
        '基金代码': code,
        '基金名称': name,
        '基金类型': fund_type,
        '年化收益率 (%)': metrics['年化收益率 (%)'],
        '年化波动率 (%)': metrics['年化波动率 (%)'],
        '夏普比率': metrics['夏普比率'],
        '管理费 (%)': fee,
        '实时估值涨跌幅 (%)': realtime_estimate,
        '基金公司': company,
        '行业分布': industry_df.to_dict('records') if not industry_df.empty else [],
        '行业集中度 (%)': top3_concentration,
        '综合评分': score,
        '筛选结果': '合格' if not reason_list else '不合格',
        '未通过原因': reason
    }
    if not reason_list:
        print(f"    √ {name} 筛选通过，综合评分: {score}", flush=True)
        return result
    else:
        print(f"    × {name} 筛选不合格: {reason}", flush=True)
        return None


if __name__ == "__main__":
    start_date_str = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')
    print(f">>> 正在筛选过去3年（截至{datetime.now().strftime('%Y-%m-%d')}）的基金，请耐心等待...", flush=True)
    fund_df = get_all_funds_from_eastmoney()
    if fund_df.empty:
        print(">>> 无法获取基金列表，程序退出。", flush=True)
    else:
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(calculate_score_and_filter, fund, start_date_str): fund for _, fund in fund_df.iterrows()}
            for future in tqdm(futures, desc="处理基金进度"):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"    × 处理基金时发生异常: {e}", flush=True)
                    traceback.print_exc()
        if results:
            final_df = pd.DataFrame(results).sort_values('综合评分', ascending=False).reset_index(drop=True)
            final_df.index = final_df.index + 1
            print("\n--- 筛选完成，推荐基金列表 ---", flush=True)
            print(final_df.drop(columns=['行业分布']).to_string(), flush=True)
            final_df.to_csv('recommended_cn_funds.csv', index=True, index_label='排名', encoding='utf-8-sig')
            print("\n>>> 推荐结果已保存至 recommended_cn_funds.csv", flush=True)
            for idx, row in final_df.iterrows():
                code = row['基金代码']
                name = row['基金名称']
                print(f"\n--- 基金 {name} ({code}) 持仓详情 ---", flush=True)
                if row['行业分布']:
                    industry_df = pd.DataFrame(row['行业分布'])
                    print(industry_df.to_string(index=False), flush=True)
                    print(f"    行业集中度（前三大行业占比）: {row['行业集中度 (%)']:.2f}%", flush=True)
                else:
                    print("    × 无持仓数据。", flush=True)
        else:
            print("\n>>> 未找到符合条件的基金，建议调整筛选条件。", flush=True)
