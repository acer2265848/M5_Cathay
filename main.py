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
sample_submission = pd.read_csv(r'data/sample_submission.csv')
#%%
#  ref:https://www.kaggle.com/fabiendaniel/elo-world
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def encode_categorical(df, cols):   
    for col in cols:
        le = LabelEncoder()
        df[col] = df[col].fillna('nan')
        df[col] = pd.Series(le.fit_transform(df[col]), index=df.index)
    return df
def prep_calendar(df):
    df = df.drop(["date", "weekday", "event_type_1", "event_type_2"], axis=1)
    df = df.assign(d = df.d.str[2:].astype(int))
    to_ordinal = ["event_name_1", "event_name_2"] 
    df[to_ordinal] = df[to_ordinal].fillna("1")
    df[to_ordinal] = OrdinalEncoder(dtype="int").fit_transform(df[to_ordinal]) + 1
    to_int8 = ["wday", "month", "snap_CA", "snap_TX", "snap_WI"] + to_ordinal
    df[to_int8] = df[to_int8].astype("int8")
    return df
#%%
sell_prices = reduce_mem_usage(sell_prices)
# memory usage: 130.5+ MB
# calendar = prep_calendar(calendar).pipe(reduce_mem_usage)
calendar = reduce_mem_usage(calendar)
# 0.12 Mb
sales = reduce_mem_usage(sales)
# 96.13 Mb
#%% plot memory usage
# df_memory = pd.DataFrame(columns=['Before','After','Data'])
# df_memory['Data'] = ['sell_prices','calendar','evaluation','validation']
# df_memory['Before'] = [208.8,0.2,452.9,446.4]
# df_memory['After'] = [130.5,0.12,96.13,95]
# df_memory = pd.melt(df_memory, id_vars='Data', var_name='Status', value_name='Memory (MB)')
# fig = plt.figure(figsize=(24,10),dpi=300)
# sns_bar = sns.barplot(x="Data", y="Memory (MB)", hue="Status", data=df_memory)
# for p in sns_bar.patches:
#     sns_bar.annotate(format(p.get_height(), '.1f')+' MB', 
#                    (p.get_x() + p.get_width() / 2., p.get_height()), 
#                    ha = 'center', va = 'center', 
#                    xytext = (0, 9), 
#                    fontsize=15,
#                    textcoords = 'offset points')
# sns_bar.set_xlabel("Data",fontsize=20)
# sns_bar.set_ylabel("Memory",fontsize=20)
# sns_bar.tick_params(labelsize=15)
# plt.setp(sns_bar.get_legend().get_texts(), fontsize='20') # for legend text
# plt.setp(sns_bar.get_legend().get_title(), fontsize='20')
# fig.suptitle("Memory reducing between before and after", fontsize=30,y=0.93)
#%%
# sales['id'].value_counts() #uuid
# sales['item_id'].value_counts() #uuid
# sales['dept_id'].value_counts() #7
# # =============================================================================
# # FOODS_3        8230
# # HOUSEHOLD_1    5320
# # HOUSEHOLD_2    5150
# # HOBBIES_1      4160
# # FOODS_2        3980
# # FOODS_1        2160
# # HOBBIES_2      1490
# # =============================================================================
# sales['cat_id'].value_counts() #3
# sales['store_id'].value_counts() #10
# sales['state_id'].value_counts() #3
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
'''del date of sale not yet.'''
# filter_noSale = []

# for store in df['store_id'].unique().tolist():
#     df2 = df.loc[(df['store_id'] == store),['id','sold']]
#     start_time = time.time()
#     for index,ids in enumerate(df2['id'].unique().tolist()):
#         if (index % 500) == 0:
#             print("%s , %s"%(store,index))
        
#         df3 = df2.loc[(df2['id']==ids)]
#         start = df3.loc[(df3['sold'] != 0)].index
#         if len(start) > 0:
#             filtered_series = df3.loc[:start[0],:].index[:-1]
#             # df2 = df2.drop(filtered_series)
#         filter_noSale.append(filtered_series)
#     end_time = time.time()
#     print("%s 執行時間：%f 秒" % (store,(end_time - start_time)))
# #%% write to pickle, otherwise too waste time
# dropNSale_toPickle = open('dropNSale.pickle','wb')
# pickle.dump(filter_noSale,dropNSale_toPickle)
# dropNSale_toPickle.close()
#%% filter no sale date.
df_back = df.copy()
df = df.drop(filter_noSale)
#%%
# Add sold features
for lag in lags:
    df[f'{lag}D_sold_lag'] = df.groupby('id')['sold'].transform(lambda x: x.shift(lag)).astype("float16")
    for day in days:
        df[f"{day}D_{lag}lag_rolling_sold_mean"] = df.groupby('id')[f'{lag}D_sold_lag'].transform(lambda x: x.rolling(day).mean()).astype("float16")
df = df.assign(lag_diff = df['28D_sold_lag']-df['7D_sold_lag'])
# del na (feature eng...)
df = df.loc[df['d'] > (max(lags) + max(days))]
# merge 3 datasets
df = df.merge(calendar, how="left", on="d")
df = df.merge(sell_prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])
df = df.drop(["wm_yr_wk"], axis=1)
df = encode_categorical(df, ["item_id", "store_id", "state_id", "dept_id", "cat_id"])
#%% write to pickle, otherwise too waste time
df_all_toPickle = open('m5_df_all.pickle','wb')
pickle.dump(df,df_all_toPickle)
df_all_toPickle.close()
#%% model
no = ['id', 'd', 'sold']
train_var = [x for x in df.columns if x not in no]        
CV = [[1802,1829],[1830,1857],[1858,1885],[1886,1913],[1914,1941]]
oof_rmse = []
for CVi in CV[-1:]:
    train_x, train_y = df.loc[df['d'] < CVi[0],train_var], df.loc[df['d'] < CVi[0],"sold"]
    valid_x = df.loc[(df['d'] > CVi[0])&(df['d'] < CVi[1]),train_var]
    valid_y = df.loc[(df['d'] > CVi[0])&(df['d'] < CVi[1]),"sold"]
    # train_x, valid_x, train_y, valid_y = train_test_split(df[train_var], df["sold"], test_size=0.3, shuffle=True, random_state=0)
    train = lgb.Dataset(train_x, label = train_y)
    valid = lgb.Dataset(valid_x, label = valid_y)

    params = {
        'metric': 'rmse',
        # 'objective': 'poisson',
        'objective': 'tweedie',
        'force_row_wise' : True,
        'learning_rate' : 0.075,
        'lambda': 0.1,
        'num_leaves': 127,
        'sub_row' : 0.75,
        'bagging_freq' : 1,
        'colsample_bytree': 0.7
    }
    
    model = lgb.train(params, 
                        train, 
                        num_boost_round = 1000, 
                        valid_sets = [valid], 
                        early_stopping_rounds = 200,
                        verbose_eval = 200)
    lgb.plot_importance(model, importance_type="gain", precision=0, height=0.5, figsize=(6, 10));
    oof_vali = model.predict(valid_x)
    oof_rmse.append(np.sqrt(metrics.mean_squared_error(valid_y, oof_vali)))
test = df.loc[df['d'] >= (d_End-max(lags)-max(days)-28)]

#%%
def testing_features(df1):
    # df1  = test.loc[(test['d'] <= day) & (test['d'] >= day - max(lags) - max(days))].copy()
    # df1_10  = df1.loc[df1['id']=='HOBBIES_1_001_CA_1']
    out = df1.groupby('id', sort=False).last()
    for lag in lags:
        out[f'{lag}D_sold_lag'] = df1.groupby('id', sort=False)['sold'].nth(-lag-1).astype("float32")
        for day in days:
            out[f"{day}D_{lag}lag_rolling_sold_mean"] = df1.groupby('id', sort=False)['sold'].nth(list(range(-lag-day, -lag))).groupby('id', sort=False).mean().astype("float32")
    
    return out.reset_index()
#%%
for i, ddd in enumerate(np.arange(d_End, d_End+d_test)):
    print(i)
    print(ddd)
    test_day = testing_features(test.loc[(test['d'] <= ddd) & (test['d'] >= (ddd - max(lags) - max(days)))])
    test.loc[test.d == ddd, ["sold"]] = model.predict(test_day[train_var])
#%%
test_sub = test.copy()
cols = sample_submission.columns
test_sub = test_sub.assign(id=test_sub['id'] + "_" + np.where(test_sub['d'] < d_End, "validation", "evaluation"),
                   F="F" + (test_sub['d'] - d_End + d_test + 1 - d_test * (test_sub['d'] >= d_End)).astype("str"))
submission = test_sub.pivot(index="id", columns="F", values="sold").reset_index()[cols]
# pandas can not fillna with float16 and int16, need to convert it to 32.
toFloat32 = ['sell_price','1D_sold_lag', '7D_1lag_rolling_sold_mean',
            '14D_1lag_rolling_sold_mean', '28D_1lag_rolling_sold_mean',
            '7D_sold_lag', '7D_7lag_rolling_sold_mean',
            '14D_7lag_rolling_sold_mean', '28D_7lag_rolling_sold_mean',
            '14D_sold_lag', '7D_14lag_rolling_sold_mean',
            '14D_14lag_rolling_sold_mean', '28D_14lag_rolling_sold_mean',
            '28D_sold_lag', '7D_28lag_rolling_sold_mean',
            '14D_28lag_rolling_sold_mean', '28D_28lag_rolling_sold_mean','lag_diff','sold']
test_sub[toFloat32] = test_sub[toFloat32].astype("float32")

toInt32 = ['wday','month','year','d','event_name_1','event_name_2','snap_CA','snap_TX','snap_WI']
test_sub[toInt32] = test_sub[toInt32].astype("int32")

submission = test_sub.pivot_table(index=['id'], columns='F')['sold'].reset_index().fillna(1)
submission = submission[cols]

submission.to_csv("submission.csv", index=False)
#%%    
# =============================================================================
# 
# 
# Below is for something test and plot
# 
# 
# =============================================================================
# df_nonZero = df.loc[(df['id']=='HOBBIES_1_001_CA_1')|(df['id']=='HOBBIES_1_001_CA_2')]
# df_nonZero = df.loc[(df['id']=='FOODS_2_154_WI_3')]
#%%
df.set_index('App').T.plot(kind='bar', stacked=True)
start_time = time.time()
df2 = df.loc[(df['store_id']=='CA_1')]
df_nonZero = df2.loc[(df2['id']=='HOBBIES_1_001_CA_1')]
df_Zero = df_nonZero.copy()
# s = s.sort_index()
for ids in df_nonZero['id'].unique().tolist():
    print(ids)
    start = df_nonZero.loc[(df_nonZero['id']==ids)&(df_nonZero['sold'] != 0)].index
    if len(start) > 0:
        filtered_series = df_nonZero.loc[(df_nonZero['id']==ids)].loc[:start[0],:].index[:-1]
        df_nonZero = df_nonZero.drop(filtered_series)
# df_nonZero_1 = df_nonZero.dropna(how='all')
# df_nonZero.loc[(df_nonZero['id']=='HOBBIES_1_001_CA_1'),['sold']].plot()
end_time = time.time()
print("執行時間：%f 秒" % (end_time - start_time))


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
#%% unmelt # df.pivot is not work on some ver. # df.pivot_table is very slow for e3_1230
# df2 = df.pivot_table(index=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
#                , columns='d')['sold'].reset_index(drop=False)