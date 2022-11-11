#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
np.random.seed(19)
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import figure

def load_data(ServicePoint):
    f=pd.read_csv('/work2/05067/nagyz/austin_energy/data/_4_exports_UT/UT_sID_'+str(ServicePoint)+'_dates_2017.01.01_through_2021.10.01.csv',index_col=False)
    f.iloc[:,0]=f.iloc[:,0].str.replace('T',' ')
    f.iloc[:,0]=f.iloc[:,0].str.replace('Z',' ')
    f.iloc[:,0]=f.iloc[:,0].str.replace('.000',' ')
    f.iloc[:,2]=f.iloc[:,2].str.replace('T',' ')
    f.iloc[:,2]=f.iloc[:,2].str.replace('Z',' ')
    f.iloc[:,2]=f.iloc[:,2].str.replace('.000',' ')
    f.iloc[:,0]= pd.to_datetime(f.iloc[:,0])
    f.iloc[:,2]= pd.to_datetime(f.iloc[:,0])
    return f
def load_graph_kwh(servicepoint):
    s=load_data(servicepoint)
    plt.plot(s[f'{servicepoint} kWh DateTime'],s[f'{servicepoint} kWh Value'],'-')
    plt.ylabel('kwh')
    plt.xlabel('year')
    plt.legend()
def yearly_graph_kwh(servicepoint,year):
    s=load_data(servicepoint)
    s["year"] = pd.to_datetime(s[f'{servicepoint} kWh DateTime']).dt.year
    plt.plot(s.loc[s['year']==year][f'{servicepoint} kWh DateTime'],s.loc[s['year']==year][f'{servicepoint} kWh Value'],'-')
    plt.xlabel('TimeLine(yearly)')
    plt.ylabel('KWH Value')
    plt.legend()
def load_graph_count(servicepoint):
    s=load_data(servicepoint)
    plt.plot(s[f'{servicepoint} count DateTime'],s[f'{servicepoint} count Value'],'-')
    plt.ylabel('count')
    plt.xlabel('year')
    plt.legend()
def monthly_graph_kwh(servicepoint,year,month):
    s=load_data(servicepoint)
    s["year"] = pd.to_datetime(s[f'{servicepoint} kWh DateTime']).dt.year
    s["month"] = pd.to_datetime(s[f'{servicepoint} kWh DateTime']).dt.month
    plt.plot(s.loc[s['year']==year].loc[s['month']==month][f'{servicepoint} kWh DateTime'],s.loc[s['year']==year].loc[s['month']==month][f'{servicepoint} kWh Value'],'-')
    plt.xlabel('TimeLine(monthly)')
    plt.ylabel('KWH Value')
    plt.legend()
def daily_graph_kwh(servicepoint,year,month,day):
    s=load_data(servicepoint)
    s["year"] = pd.to_datetime(s[f'{servicepoint} kWh DateTime']).dt.year
    s["month"] = pd.to_datetime(s[f'{servicepoint} kWh DateTime']).dt.month
    s["day"] = pd.to_datetime(s[f'{servicepoint} kWh DateTime']).dt.day
    plt.plot(s.loc[s['year']==year].loc[s['month']==month].loc[s['day']==day][f'{servicepoint} kWh DateTime'],s.loc[s['year']==year].loc[s['month']==month].loc[s['day']==day][f'{servicepoint} kWh Value'],'-')
    plt.xlabel('TimeLine(daily)')
    plt.ylabel('KWH Value')
    plt.legend()
def load_graph_kwh_and_count(servicepoint):
    s=load_data(servicepoint)
    plt.plot(s[f'{servicepoint} kWh DateTime'],s[f'{servicepoint} kWh Value'],'-')
    plt.plot(s[f'{servicepoint} kWh DateTime'],s[f'{servicepoint} count Value'],'-')
    plt.ylabel('kwh')
    plt.xlabel('year')
    plt.legend()
def yearly_graph_kwh_and_count(servicepoint,year):
    s=load_data(servicepoint)
    s["year"] = pd.to_datetime(s[f'{servicepoint} kWh DateTime']).dt.year
    plt.plot(s.loc[s['year']==year][f'{servicepoint} kWh DateTime'],s.loc[s['year']==year][f'{servicepoint} kWh Value'],'-')
    plt.plot(s.loc[s['year']==year][f'{servicepoint} kWh DateTime'],s.loc[s['year']==year][f'{servicepoint} count Value'],'-')
    plt.xlabel('TimeLine(yearly)')
    plt.ylabel('KWH Value')
    plt.legend()
def monthly_graph_kwh_and_count(servicepoint,year,month):
    s=load_data(servicepoint)
    s["year"] = pd.to_datetime(s[f'{servicepoint} kWh DateTime']).dt.year
    s["month"] = pd.to_datetime(s[f'{servicepoint} kWh DateTime']).dt.month
    plt.plot(s.loc[s['year']==year].loc[s['month']==month][f'{servicepoint} kWh DateTime'],s.loc[s['year']==year].loc[s['month']==month][f'{servicepoint} kWh Value'],'-')
    plt.plot(s.loc[s['year']==year].loc[s['month']==month][f'{servicepoint} kWh DateTime'],s.loc[s['year']==year].loc[s['month']==month][f'{servicepoint} count Value'],'-')
    plt.xlabel('TimeLine(monthly)')
    plt.ylabel('KWH Value')
    plt.legend()
def daily_graph_kwh_and_count(servicepoint,year,month,day):
    s=load_data(servicepoint)
    s["year"] = pd.to_datetime(s[f'{servicepoint} kWh DateTime']).dt.year
    s["month"] = pd.to_datetime(s[f'{servicepoint} kWh DateTime']).dt.month
    s["day"] = pd.to_datetime(s[f'{servicepoint} kWh DateTime']).dt.day
    plt.plot(s.loc[s['year']==year].loc[s['month']==month].loc[s['day']==day][f'{servicepoint} kWh DateTime'],s.loc[s['year']==year].loc[s['month']==month].loc[s['day']==day][f'{servicepoint} kWh Value'],'-')
    plt.plot(s.loc[s['year']==year].loc[s['month']==month].loc[s['day']==day][f'{servicepoint} kWh DateTime'],s.loc[s['year']==year].loc[s['month']==month].loc[s['day']==day][f'{servicepoint} count Value'],'-')
    plt.xlabel('TimeLine(daily)')
    plt.ylabel('KWH Value')
    plt.legend()







