!pip install pysentiment2
import pysentiment2 as ps
import pandas as pd
import numpy as np
import statsmodels.tsa.api as smtsa
import datetime as dt

#processing news data
df = pd.read_csv('C:/Users/86183/Desktop/Numtech/abcnews-date-text.csv')
df['headline_text'] = df['headline_text'].map(str.lower)
df['publish_date'] = pd.to_datetime(df['publish_date'],format='%Y%m%d')
df = df[df['publish_date'] >= dt.datetime(2013,1,1)]
df.reset_index(drop=True,inplace=True)
keywords = ['oil', 'crude', 'gas', 'energy', 'petrol', 'fuel']
def filkw(text):
    result = False
    for kw in keywords:
        if kw in text:
            return True
    return 0
df['score0'] = df['headline_text'].map(filkw)
def text2score(row):
    if row['score0'] == 0:
        return 0
    else:
        text = row['headline_text']
        lm = ps.LM()
        tokens = lm.tokenize(text)
        score = lm.get_score(tokens)
        score_p = score['Positive']
        score_n = score['Negative']
        if score_p > score_n:
            return 1
        elif score_p == score_n:
            return 0
        else:
            return -1
    
df['score_final'] = df.apply(text2score,axis=1)
day_new = df.groupby('publish_date').mean()
day_new.to_csv('daily_news.csv')

#constructed realized volatility
df13_16 = pd.read_csv('LIGHT.CMDUSD_Candlestick_5_M_ASK_01.01.2013-01.01.2016.csv')
df16_17 = pd.read_csv('LIGHT.CMDUSD_Candlestick_5_M_ASK_02.01.2016-30.12.2017.csv')
df17_20 = pd.read_csv('LIGHT.CMDUSD_Candlestick_5_M_ASK_31.12.2017-31.12.2020.csv')
df = pd.concat([df13_16,df16_17,df17_20])
df['close1'] = df['Close'].shift(1)
df['rt'] = np.log(df['close1'] / df['Close'])
df.dropna(inplace=True)
def stripd(text):
    timelst = text.split()
    dt = timelst[0]
    return dt
df['day'] = pd.to_datetime(df['Local time'].map(stripd),format='%d.%m.%Y')

def dplus(rt):
    if rt >= 0:
        return rt**2
    return 0
def dminus(rt):
    if rt < 0:
        return rt**2
    return 0

df['dp'] = df['rt'].map(dplus)
df['dn'] = df['rt'].map(dminus)
df['rt2'] = df['rt'] ** 2
dfday = df.groupby('day')[['rt2','dp','dn']].sum()
rv5 = dfday[dfday['rt2'] != 0]
rv5['w'] = (rv5['rt2'].rolling(5).sum()) / 5
rv5['wp'] = (rv5['dp'].rolling(5).sum()) / 5
rv5['wn'] = (rv5['dn'].rolling(5).sum()) / 5
rv5['m'] = (rv5['rt2'].rolling(22).sum()) / 22
rv5['mp'] = (rv5['dp'].rolling(22).sum()) / 22
rv5['mn'] = (rv5['dn'].rolling(22).sum()) / 22
rv5.to_csv('rv5_oil_HAR.csv')

#process & merge all data set
df_cr = pd.read_csv('^cry_d.csv')
df_ov = pd.read_csv('^OVX.csv')
df_vi = pd.read_csv('^VIX.csv')
df_us = pd.read_csv('usd_i_d.csv')
df_go = pd.read_csv('XAUUSD_Candlestick_1_D_ASK_01.01.2013-31.12.2020.csv')

df_lst = [df_cr,df_ov,df_vi,df_us]
for df in df_lst: 
    df['dt'] = pd.to_datetime(df_cr['Date'],format='%d/%m/%Y')

df_go['dt'] = pd.to_datetime(df_go['Local time'].map(stripd),format='%d.%m.%Y')

df_all = pd.merge(df_cr[['dt','Close']],df_ov[['dt','Close']],on='dt',how='inner',suffixes=['CRY','OVX'])
df_all1 = pd.merge(df_all,df_vi[['dt','Close']],on='dt',how='inner',suffixes=['','VIX'])
df_all2 = pd.merge(df_all1,df_us[['dt','Close']],on='dt',how='inner',suffixes=['VIX','USD'])
df_go['CloseGOLD'] = df_go['Close']
df_all3 = pd.merge(df_all2,df_go[['dt','CloseGOLD']],on='dt',how='inner')
all_in = pd.read_csv('ALL_in_one(rolling).csv',parse_dates=['daytime'])
all_in1 = pd.merge(all_in,df_all3,left_on='daytime',right_on='dt',how='inner')
all_in1.dropna(inplace=True)
all_in1.to_csv('final_data.csv')