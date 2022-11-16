#!/usr/bin/env python
# coding: utf-8

# In[2]:
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
np.random.seed(19)
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import figure

def length(ZipCode,Type):
    f=len(df.loc[df['ZipCode']==ZipCode].loc[df['Type']==Type])
    return f


# In[3]:


def prav3(ZipCode,Type):
    dataframes=[]
    for i in np.arange(len(df.loc[df['ZipCode']==ZipCode].loc[df['Type']==Type])):
        df2=df.loc[df['ZipCode']==ZipCode].loc[df['Type']==Type]
        dataframes.append(load_data(df2.iloc[i]['Service Point']))
    r=pd.concat([dataframes[0][dataframes[0].columns[1]],dataframes[1][dataframes[1].columns[1]],dataframes[2][dataframes[2].columns[1]]]).groupby(level=0).mean()
    return r
def prav2(ZipCode,Type):
    dataframes=[]
    for i in np.arange(len(df.loc[df['ZipCode']==ZipCode].loc[df['Type']==Type])):
        df2=df.loc[df['ZipCode']==ZipCode].loc[df['Type']==Type]
        dataframes.append(load_data(df2.iloc[i]['Service Point']))
    s=pd.concat([dataframes[0][dataframes[0].columns[1]],dataframes[1][dataframes[1].columns[1]]]).groupby(level=0).mean()
    return s
def prav1(ZipCode,Type):
    dataframes=[]
    for i in np.arange(len(df.loc[df['ZipCode']==ZipCode].loc[df['Type']==Type])):
        df2=df.loc[df['ZipCode']==ZipCode].loc[df['Type']==Type]
        dataframes.append(load_data(df2.iloc[i]['Service Point']))
    q=pd.concat([dataframes[0][dataframes[0].columns[1]]]).groupby(level=0).mean()
    return q


# In[4]:


def dinostorm(ZipCode,Type):
    if length(ZipCode,Type)==3:
        return prav3(ZipCode,Type)
    elif length(ZipCode,Type)==2:
         return prav2(ZipCode,Type)
    elif length(ZipCode,Type)==1:
        return prav1(ZipCode,Type)
         


# In[5]:


def plot_aggraph(ZipCode,Type):
    h=dinostorm(ZipCode,Type)
    h1=pd.DataFrame(h)
    h2=h1.set_index(dati[dati.columns[0]][:h1.shape[0]])
    from sklearn.preprocessing import MinMaxScaler
    Scaler=MinMaxScaler()
    h2 = h2.rename(columns={'-1529 kWh DateTime': 'TIME STAMP', 0:'KWH VALUE'})
    h3=Scaler.fit_transform(h2)
    plt.plot(h2.index,h3,'-',label=Type)
    plt.xlabel('TimeLine(daily)')
    plt.ylabel('KWH Value')
    plt.legend()
    plt.show
    plt.title(ZipCode)


# In[6]:


def heatmap(ZipCode,Type):
    h=dinostorm(ZipCode,Type)
    h1=pd.DataFrame(h)
    h2=h1.set_index(dati[dati.columns[0]][:h1.shape[0]])
    h2 = h2.rename(columns={ 0:'KWH VALUE'})
    h3=h2.resample('D').sum()
    h4=Scaler.fit_transform(h3)
    h4=pd.DataFrame(h4)
    h6 = h4.rename(columns={0:Type})
    h7=h6.set_index(h3.index)
    return h7


# In[7]:


def finalheatmap(zip_code):
    zip_code = zip_code
    data_list = []
    for building_type in df[df['ZipCode'] == zip_code]['Type'].unique():
         data = heatmap(zip_code, building_type)
         data= data.reset_index(drop=False)
         data.columns = ['timestamp','value']
         data['building_type'] = building_type
         data_list.append(data)
    data = pd.concat(data_list, ignore_index = True)
    data = data.pivot(index='building_type',columns='timestamp',values='value')
    return data


# In[ ]:




