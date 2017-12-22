import numpy as np
import json
import urllib.request as Request
import pandas as pd
url_templete='http://q.stock.sohu.com/hisHq?code=%s_%s&start=%s&end=%s&stat=1&order=D&period=d&callback=historySearchHandler&rt=jsonp'
cols=['date','open','close','price_change','p_change_rate','low','high','volume','amount','turnover']

def spell_url(domain, code,start,end):
    url=url_templete%(domain,code,start,end)
    return url

def get_market_data(code,start,end,is_index=False):
    if not is_index:
        domain='cn'
    else:
        domain='zs'
    url = spell_url(domain,code,start,end)
    request = Request.Request(url)
    response = Request.urlopen(request)
    lines = response.read()
    lines = lines.decode('gb2312')
    js = json.loads(lines[22:-3])
    df = pd.DataFrame(js['hq'], columns=cols)
    df = df.set_index('date')
    df = df.sort_index(ascending = False)
    if is_index:
        df=df[cols[1:-1]]
    df = df.applymap(lambda x:x.strip('%'))
    df[df==''] = 0
    for col in df.columns:
        df[col]=df[col].astype(float)
    return df

def get_hist_data(code,start,end,with_index=True):
    if with_index:
        df = get_market_data(code,start,end,False)
        df_index = get_market_data('399001', start,end,True)
        for col in cols[1:-1]:
            col_ = 'index_' + col
            df[col_]=df_index[col]
        return df
    return get_market_data(code,start,end,False)
