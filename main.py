# coding=UTF-8

import requests
import json
import pandas
import time
import os

class Djeva():
    _url = 'https://danjuanfunds.com/djapi/index_eva/dj'
    _headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.64 Safari/537.36 Edg/101.0.1210.53'
    }
    _source_dir = 'source'
    _csv_dir = 'csv'

    def __init__(self, fund_codes):
        self.fund_codes = fund_codes
        self.date = time.strftime("%Y-%m-%d", time.localtime())
        self.all_items = []

        # 创建目录
        if not os.path.exists(self._source_dir):
            os.makedirs(self._source_dir)
        if not os.path.exists(self._csv_dir):
            os.makedirs(self._csv_dir)

        # 只调用一次 API 获取所有数据
        all_data = self._fetch()
        if not all_data:
            print("未能获取到任何数据，请检查网络连接。")
            return
            
        # 遍历所有数据，筛选出你需要的基金
        items = all_data.get('data', {}).get('items', [])
        for item in items:
            if item.get('index_code') in self.fund_codes or item.get('name') in self.fund_codes:
                self.all_items.append(item)

        if not self.all_items:
            print("未能找到你所指定的基金，请检查基金代码是否正确。")
            return

        # 导出数据
        csv_file = os.path.join(self._csv_dir, self.date)
        self.dump('csv', filename=csv_file, data=self.all_items)

    def dump(self, dtype: str, **kw):
        dump_ops = {
            'csv': self._dump_csv
        }
        return dump_ops[dtype](**kw)

    def _dump_csv(self, filename, data):
        filename = filename + '.csv'
        pandas.DataFrame(data).to_csv(filename, index=None, encoding='utf-8-sig')
        print(f"你所需要的基金数据已保存到 {filename}")
        return filename

    def _fetch(self):
        try:
            print(f"正在获取所有基金数据...")
            response = requests.get(self._url, headers=self._headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"请求数据失败: {e}")
            return None

def main():
    # 在这里填写你的基金代码或基金名称
    fund_codes_to_fetch = ['110022', '161005', 'SZ399393', '中证1000']
    Djeva(fund_codes_to_fetch)

if __name__ == '__main__':
    main()
