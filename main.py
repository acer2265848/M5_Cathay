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
#%%
with open(r'm5_df_all.pickle','rb') as file:
    m5_df_all = pickle.load(file)
#%% read files
sell_prices = pd.read_csv(r'data/sell_prices.csv')
# memory usage: 208.8+ MB
calendar = pd.read_csv(r'data/calendar.csv')
# memory usage: 215.5+ KB
sales_train_evaluation = pd.read_csv(r'data/sales_train_evaluation.csv')
# memory usage: 452.9+ MB
sales_train_validation = pd.read_csv(r'data/sales_train_validation.csv')
# memory usage: 446.4+ MB
calendar['date'] = pd.to_datetime(calendar['date'])
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
#%%
def encode_categorical(df, cols):   
    for col in cols:
        le = LabelEncoder()
        df[col] = df[col].fillna('nan')
        df[col] = pd.Series(le.fit_transform(df[col]), index=df.index)
    return df
sell_prices = encode_categorical(sell_prices, ["item_id", "store_id"]).pipe(reduce_mem_usage)
# memory usage: 130.5+ MB
calendar = encode_categorical(calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]).pipe(reduce_mem_usage)
# 0.12 Mb
sales_train_evaluation = encode_categorical(sales_train_evaluation, ["item_id", "dept_id", "cat_id", "store_id", "state_id"],).pipe(reduce_mem_usage)
# 96.13 Mb
sales_train_validation = encode_categorical(sales_train_validation, ["item_id", "dept_id", "cat_id", "store_id", "state_id"],).pipe(reduce_mem_usage)
# 95.00 Mb
sss  = encode_categorical(calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"])
d_store_id = dict(zip(df_all.store_id.cat.codes, df_all.store_id))
# sell_prices = reduce_mem_usage(sell_prices)
# # memory usage: 130.5+ MB
# calendar = reduce_mem_usage(calendar)
# # 0.12 Mb
# sales_train_evaluation = reduce_mem_usage(sales_train_evaluation)
# # 96.13 Mb
# sales_train_validation = reduce_mem_usage(sales_train_validation)
# # 95.00 Mb
#%%
for d in range(1942,1970):
    col = 'd_' + str(d)
    sales_train_evaluation[col] = 0
    sales_train_evaluation[col] = sales_train_evaluation[col].astype(np.int16)
for d in range(1914,1942):
    col = 'd_' + str(d)
    sales_train_validation[col] = 0
    sales_train_validation[col] = sales_train_validation[col].astype(np.int16)
#%% plot memory usage
df_memory = pd.DataFrame(columns=['Before','After','Data'])
df_memory['Data'] = ['sell_prices','calendar','evaluation','validation']
df_memory['Before'] = [208.8,0.2,452.9,446.4]
df_memory['After'] = [130.5,0.12,96.13,95]
df_memory = pd.melt(df_memory, id_vars='Data', var_name='Status', value_name='Memory (MB)')
fig = plt.figure(figsize=(24,10),dpi=300)
sns_bar = sns.barplot(x="Data", y="Memory (MB)", hue="Status", data=df_memory)
for p in sns_bar.patches:
    sns_bar.annotate(format(p.get_height(), '.1f')+' MB', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   fontsize=15,
                   textcoords = 'offset points')
sns_bar.set_xlabel("Data",fontsize=20)
sns_bar.set_ylabel("Memory",fontsize=20)
sns_bar.tick_params(labelsize=15)
plt.setp(sns_bar.get_legend().get_texts(), fontsize='20') # for legend text
plt.setp(sns_bar.get_legend().get_title(), fontsize='20')
fig.suptitle("Memory reducing between before and after", fontsize=30,y=0.93)
#%%
sales_train_evaluation['id'].value_counts() #uuid
sales_train_evaluation['item_id'].value_counts() #uuid
sales_train_validation['dept_id'].value_counts() #7
# =============================================================================
# FOODS_3        8230
# HOUSEHOLD_1    5320
# HOUSEHOLD_2    5150
# HOBBIES_1      4160
# FOODS_2        3980
# FOODS_1        2160
# HOBBIES_2      1490
# =============================================================================
sales_train_evaluation['cat_id'].value_counts() #3
sales_train_evaluation['store_id'].value_counts() #10
sales_train_evaluation['state_id'].value_counts() #3
#%%
sales_train_evaluation_20 = sales_train_evaluation.head(30)
sales_train_validation_20 = sales_train_validation.head(30)

#%%
def plot_density(plot_list):
    fig = plt.figure(figsize=(10,10))
    sns.distplot(plot_list, hist = True,bins=20, kde = True,color="#A5DEE4",
                     kde_kws = {'shade': True, 'linewidth': 3},)    
    # Plot formatting
    plt.legend(loc='upper right',prop={'size': 16})
    plt.title('Distribution of sell price',fontsize=30)
    plt.xlabel('sell_price',fontsize=25)
    plt.ylabel('Density',fontsize=25)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.grid()
#%%
group = sell_prices.groupby(['year','date','state_id','store_id'], as_index=False)['sold'].sum().dropna()
#%%
fig = plt.figure(figsize=(24,10),dpi=300)
plt.subplot(2,2,1)
sns.histplot(sales_train_validation["dept_id"], binwidth=13)
plt.subplot(2,2,2)
sns.histplot(sales_train_validation["cat_id"])
plt.subplot(2,2,3)
sns.boxplot(x="store_id",y="sell_price",data=sell_prices)
# sns.histplot(sales_train_validation["store_id"])
plt.subplot(2,2,4)
sns.histplot(sales_train_validation["state_id"])
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
plot_density(sell_prices['sell_price'].tolist())
#%% melt data
df = pd.melt(sales_train_evaluation, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
             var_name='d', value_name='sold').dropna()
#%%
df_all = pd.merge(df, calendar, on='d', how='left')
df_all = pd.merge(df_all, sell_prices, on=['store_id','item_id','wm_yr_wk'], how='left') 
#%%
# df_all_10 = df_all.head(10)
# df_all_10.to_csv(r"data_test\df_all_10.csv",index=False)
# df_10 = df.head(10)
# df_10.to_csv(r"data_test\df_10.csv",index=False)
# calendar_10 = calendar.head(10)
# calendar_10.to_csv(r"data_test\calendar_10.csv",index=False)
# sell_prices_10 = sell_prices.head(10)
# sell_prices_10.to_csv(r"data_test\sell_prices_10.csv",index=False)
#%%
df_all['iteam_sold_mean'] = df_all.groupby('item_id')['sold'].transform('mean').astype(np.float16)
df_all['state_sold_mean'] = df_all.groupby('state_id')['sold'].transform('mean').astype(np.float16)
df_all['store_sold_mean'] = df_all.groupby('store_id')['sold'].transform('mean').astype(np.float16)
df_all['cat_sold_mean'] = df_all.groupby('cat_id')['sold'].transform('mean').astype(np.float16)
df_all['dept_sold_mean'] = df_all.groupby('dept_id')['sold'].transform('mean').astype(np.float16)
df_all['cat_dept_sold_mean'] = df_all.groupby(['cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)
df_all['store_item_sold_mean'] = df_all.groupby(['store_id','item_id'])['sold'].transform('mean').astype(np.float16)
df_all['cat_item_sold_mean'] = df_all.groupby(['cat_id','item_id'])['sold'].transform('mean').astype(np.float16)
df_all['dept_item_sold_mean'] = df_all.groupby(['dept_id','item_id'])['sold'].transform('mean').astype(np.float16)
df_all['state_store_sold_mean'] = df_all.groupby(['state_id','store_id'])['sold'].transform('mean').astype(np.float16)
df_all['state_store_cat_sold_mean'] = df_all.groupby(['state_id','store_id','cat_id'])['sold'].transform('mean').astype(np.float16)
df_all['store_cat_dept_sold_mean'] = df_all.groupby(['store_id','cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)
#%% rolling_sold_mean 
days = [7, 30]
for day in days:
    df_all[f"{day}D_rolling_sold_mean"] = df_all.groupby(['id', 'item_id', 'dept_id', 'cat_id', 
                                                            'store_id', 'state_id'])\
                                                          ['sold'].transform(lambda x: x.rolling(window=day).mean()).astype(np.float16)
    df_all[f"{day}D_rolling_sold_max"] = df_all.groupby(['id', 'item_id', 'dept_id', 'cat_id', 
                                                            'store_id', 'state_id'])\
                                                          ['sold'].transform(lambda x: x.rolling(window=day).max()).astype(np.float16)
    df_all[f"{day}D_rolling_sold_min"] = df_all.groupby(['id', 'item_id', 'dept_id', 'cat_id', 
                                                            'store_id', 'state_id'])\
                                                          ['sold'].transform(lambda x: x.rolling(window=day).min()).astype(np.float16)
lags = [1,7,15,30]
for lag in lags:
    df_all[f'{lag}D_sold_lag'] = df_all.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],as_index=False)['sold'].shift(lag).astype(np.float16)                                                          
#%% filter lag days
df_all['d_num'] = df_all['d'].apply(lambda x: x[2:]).astype(np.float16)

df_all = df_all.loc[df_all['d_num']>=max(lags)]
#%% write to pickle, otherwise too waste time
df_all_toPickle = open('m5_df_all.pickle','wb')
pickle.dump(df_all,df_all_toPickle)
df_all_toPickle.close()
#%% modeling
valid = df_all.loc[(df_all['d_num']>=1914) & (df_all['d_num']<1942),['id','d_num','sold']]
test = df_all.loc[df_all['d_num']>=1942,['id','d_num','sold']]
eval_preds = test['sold']
valid_preds = valid['sold']
#%%
#Get the store ids
# sell_prices['store_id'] = sell_prices['store_id'].astype("category")
stores = sell_prices["store_id"].unique().tolist()
for store in stores:
    df = df_all[df_all['store_id']==store]
    
    #Split the data
    X_train, y_train = df[df['d']<1914].drop('sold',axis=1), df[df['d']<1914]['sold']
    X_valid, y_valid = df[(df['d']>=1914) & (df['d']<1942)].drop('sold',axis=1), df[(df['d']>=1914) & (df['d']<1942)]['sold']
    X_test = df[df['d']>=1942].drop('sold',axis=1)
    
    #Train and validate
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=8,
        num_leaves=50,
        min_child_weight=300
    )
    print('*****Prediction for Store: {}*****'.format(d_store_id[store]))
    model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_valid,y_valid)],
             eval_metric='rmse', verbose=20, early_stopping_rounds=20)
    valid_preds[X_valid.index] = model.predict(X_valid)
    eval_preds[X_test.index] = model.predict(X_test)
    filename = 'model'+str(d_store_id[store])+'.pkl'
    # save model
    joblib.dump(model, filename)
    del model, X_train, y_train, X_valid, y_valid
    gc.collect()

#%% snap observation (plot)
fig, ax = plt.subplots(3, 1, constrained_layout=True,figsize=(15,8), dpi=300)
date_before = '2016-01-01'
date_after = '2016-03-01'
ax[0].plot(df_all.loc[(date_before<= df_all['date'])& (df_all['date']<= date_after),'date']
           ,df_all.loc[(date_before<= df_all['date'])& (df_all['date']<= date_after),'snap_CA'].tolist()
           ,'b.',linewidth=0.8, alpha=0.5,label='CA',markersize=2.5)
ax[0].plot(df_all.loc[(date_before<= df_all['date'])& (df_all['date']<= date_after),'date']
           ,df_all.loc[(date_before<= df_all['date'])& (df_all['date']<= date_after),'snap_CA'].tolist()
           ,'b',linewidth=0.8, alpha=0.5,label='CA')
ax2 = ax[0].twinx()
ax2.set_ylabel('sum of squared ITU',size = 20)
ax2.plot(df_all.loc[(date_before<= df_all['date'])& (df_all['date']<= date_after)&(df_all['state_id']=='CA'),'date']
           ,df_all.loc[(date_before<= df_all['date'])& (df_all['date']<= date_after)&(df_all['state_id']=='CA'),'sold'].tolist()
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
df2 = df.pivot_table(index=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
               , columns='d')['sold'].reset_index(drop=False)