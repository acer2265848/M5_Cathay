# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:19:07 2021

@author: HF535UH
"""
#%% package
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import seaborn as sns
#%% read files
sell_prices = pd.read_csv(r'data/sell_prices.csv')
calendar = pd.read_csv(r'data/calendar.csv')
sales_train_evaluation = pd.read_csv(r'data/sales_train_evaluation.csv')
sales_train_validation = pd.read_csv(r'data/sales_train_validation.csv')
#%%
def plot_density(plot_list):
    lable_list = 'distribution of list'
    
    fig = plt.figure(figsize=(10,10))
    sns.distplot(plot_list, hist = True,bins=20, kde = True,color="#A5DEE4",
                     kde_kws = {'shade': True, 'linewidth': 3},)    
    # Plot formatting
    plt.legend(loc='upper right',prop={'size': 16})
    plt.title('Distribution of error counts',fontsize=30)
    plt.xlabel('sell_price',fontsize=25)
    plt.ylabel('Density',fontsize=25)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.grid()
plot_density(sell_prices['sell_price'].tolist())
