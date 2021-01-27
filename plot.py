# -*- coding: utf-8 -*-

#%% package
import pandas as pd
import numpy as np
# import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import pickle
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from fbprophet import Prophet
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
#%% filter non sale date
with open(r'dropNSale.pickle','rb') as file:
    dropNSale = pickle.load(file)
filter_noSale = [item for sublist in dropNSale for item in sublist]
#%% read files
sell_prices = pd.read_csv(r'data/sell_prices.csv')
# memory usage: 208.8+ MB
calendar = pd.read_csv(r'data/calendar.csv')
# memory usage: 215.5+ KB
sales = pd.read_csv(r'data/sales_train_evaluation.csv')
# memory usage: 452.9+ MB
calendar['date'] = pd.to_datetime(calendar['date'])

#%%
d_End = 1942 
d_test = 28 #(1942+28)
lags = [1, 7, 14, 28]
days = [7, 14, 28]
#%%
df = sales.copy() 
# melt
df = df.assign(id=df.id.str.replace("_evaluation", ""))
df = df.reindex(columns=df.columns.tolist() + ["d_" + str(d_End + i) for i in range(d_test)])
df = df.melt(id_vars=["id", "item_id", "store_id", "state_id", "dept_id", "cat_id"], var_name='d', value_name='sold')
df = df.assign(d=df['d'].str[2:].astype("int16"),
               sold=df['sold'].astype("float16"))

#%%
# Add sold features
# merge 3 datasets
df = df.merge(calendar, how="left", on="d")
df = df.merge(sell_prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])
df = df.drop(["wm_yr_wk"], axis=1)
#%%    
# =============================================================================
# Below is for something test and plot
# 
# 
# =============================================================================
# df_nonZero = df.loc[(df['id']=='HOBBIES_1_001_CA_1')|(df['id']=='HOBBIES_1_001_CA_2')]
# df_nonZero = df.loc[(df['id']=='FOODS_2_154_WI_3')]
#%%
df.set_index('App').T.plot(kind='bar', stacked=True)


#%%
fig, ax = plt.subplots(2, 1, constrained_layout=True,figsize=(15,8), dpi=300)

ax[0].plot(df_Zero['d']
           ,df_Zero['sold']
           ,'b',linewidth=0.8, alpha=0.5,label='CA',markersize=2.5)
# ax[0].plot(df_all.loc[(date_before<= df_all['date'])& (df_all['date']<= date_after),'date']
#            ,df_all.loc[(date_before<= df_all['date'])& (df_all['date']<= date_after),'snap_CA'].tolist()
#            ,'b',linewidth=0.8, alpha=0.5,label='CA')
ax[0].grid(b=True, which='major', color='teal', linestyle='-',linewidth=0.5, alpha=0.3,axis='x')
ax[0].set_ylabel('sold',size = 15)
ax[0].set_xlabel('D',size = 20)
ax[0].set_title('Origin',size = 20)

ax[1].plot(df_nonZero['d']
           ,df_nonZero['sold']
           ,'b',linewidth=0.8, alpha=0.5,label='CA',markersize=2.5)
# ax[0].plot(df_all.loc[(date_before<= df_all['date'])& (df_all['date']<= date_after),'date']
#            ,df_all.loc[(date_before<= df_all['date'])& (df_all['date']<= date_after),'snap_CA'].tolist()
#            ,'b',linewidth=0.8, alpha=0.5,label='CA')
ax[1].grid(b=True, which='major', color='teal', linestyle='-',linewidth=0.5, alpha=0.3,axis='x')
ax[1].set_ylabel('sold',size = 15)
ax[1].set_xlabel('D',size = 20)
ax[1].set_title('Deletes date of unsold items',size = 20)
fig.suptitle("Copmarsion of delete date (HOBBIES_1_001_CA_2)", fontsize=30,y=1.07)


#%%
def plot_density(plot_list):
    fig = plt.figure(figsize=(15,15),dpi=300)
    sns.distplot(plot_list, hist = True,bins=50, kde = True,color="#A5DEE4",
                      kde_kws = {'shade': True, 'linewidth': 3},)    
    # Plot formatting
    # sns.histplot(plot_list)

    # plt.legend(loc='upper right',prop={'size': 16})
    plt.title('Distribution of sold',fontsize=30)
    plt.xlabel('sold',fontsize=25)
    plt.ylabel('Density',fontsize=25)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.grid()
#%%
group = sell_prices.groupby(['year','date','state_id','store_id'], as_index=False)['sold'].sum().dropna()
#%%
fig = plt.figure(figsize=(24,10),dpi=300)
plt.subplot(2,2,1)
sns.histplot(sales["dept_id"], binwidth=13)
plt.subplot(2,2,2)
sns.histplot(sales["cat_id"])
plt.subplot(2,2,3)
sns.boxplot(x="store_id",y="sell_price",data=sell_prices)
# sns.histplot(sales_train_validation["store_id"])
plt.subplot(2,2,4)
sns.histplot(sales["state_id"])
fig.suptitle("Histplot for categories", fontsize=30,y=0.93)
#%%
# CA
group_CA = calendar.groupby(['month', 'year'], as_index=False)['snap_CA'].sum().dropna()
group_CA['date'] = group_CA['year'].astype('str')+'-'+group_CA['month'].astype('str')
group_CA['date'] = pd.to_datetime(group_CA['date'])
# TX
group_TX = calendar.groupby(['month', 'year'], as_index=False)['snap_TX'].sum().dropna()
group_TX['date'] = group_TX['year'].astype('str')+'-'+group_TX['month'].astype('str')
group_TX['date'] = pd.to_datetime(group_TX['date'])
# WI
group_WI = calendar.groupby(['month', 'year'], as_index=False)['snap_WI'].sum().dropna()
group_WI['date'] = group_WI['year'].astype('str')+'-'+group_WI['month'].astype('str')
group_WI['date'] = pd.to_datetime(group_WI['date'])
# plt.show()
#%%
plot_density(df['sold'].tolist())
vv = df['sold'].dropna().tolist()
vv2 =  list(map(int, vv))
sns.histplot(vv2)
plot_density(df['sold'].dropna().tolist())

#%%
# df_all_10 = df_all.head(10)
# df_all_10.to_csv(r"data_test\df_all_10.csv",index=False)
# df_10 = df.head(10)
# df_10.to_csv(r"data_test\df_10.csv",index=False)
# calendar_10 = calendar.head(10)
# calendar_10.to_csv(r"data_test\calendar_10.csv",index=False)
# sell_prices_10 = sell_prices.head(10)
# sell_prices_10.to_csv(r"data_test\sell_prices_10.csv",index=False)
#%% snap observation (plot)
fig, ax = plt.subplots(3, 1, constrained_layout=True,figsize=(15,8), dpi=300)
date_before = '2016-01-01'
date_after = '2016-03-01'
ax[0].plot(calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after),'date']
           ,calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after),'snap_CA'].tolist()
           ,'b.',linewidth=0.8, alpha=0.5,label='CA',markersize=2.5)
ax[0].plot(calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after),'date']
           ,calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after),'snap_CA'].tolist()
           ,'b',linewidth=0.8, alpha=0.5,label='CA')
ax2 = ax[0].twinx()
ax2.set_ylabel('sum of squared ITU',size = 20)
ax2.plot(calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after)&(calendar['state_id']=='CA'),'date']
           ,calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after)&(calendar['state_id']=='CA'),'sold'].tolist()
           ,'g-',linewidth=0.8, alpha=0.5,label='CA')
ax2.tick_params(axis='y', labelcolor='b')
ax2.set_ylabel('sold_CA',size = 20,color="b")
ax[0].grid(b=True, which='major', color='teal', linestyle='-',linewidth=0.5, alpha=0.3,axis='x')
ax[0].set_ylabel('Whether SNAP?',size = 15)
ax[0].set_xlabel('Date',size = 20)
ax[0].set_title('CA_snap',size = 20)

ax[1].plot(calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after),'date']
           ,calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after),'snap_TX']
           ,'b.',linewidth=0.8, alpha=0.5,label='TX',markersize=2.5)
ax[1].plot(calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after),'date']
           ,calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after),'snap_TX']
           ,'b',linewidth=0.8, alpha=0.5,label='TX')
ax[1].grid(b=True, which='major', color='teal', linestyle='-',linewidth=0.5, alpha=0.3,axis='x')
ax[1].set_ylabel('Whether SNAP?',size = 15)
ax[1].set_xlabel('Date',size = 20)
ax[1].set_title('TX_snap',size = 20)

ax[2].plot(calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after),'date']
           ,calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after),'snap_WI']
           ,'b.',linewidth=0.8, alpha=0.5,label='WI',markersize=2.5)
ax[2].plot(calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after),'date']
           ,calendar.loc[(date_before<= calendar['date'])& (calendar['date']<= date_after),'snap_WI']
           ,'b',linewidth=0.8, alpha=0.5,label='WI')
ax[2].grid(b=True, which='major', color='teal', linestyle='-',linewidth=0.5, alpha=0.3,axis='x')
ax[2].set_ylabel('Whether SNAP?',size = 15)
ax[2].set_xlabel('Date',size = 20)
ax[2].set_title('WI_snap',size = 20)
#%% unmelt # df.pivot is not work on some ver. # df.pivot_table is very slow for e3_1230
# df2 = df.pivot_table(index=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
#                , columns='d')['sold'].reset_index(drop=False)