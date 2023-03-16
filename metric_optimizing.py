
# coding: utf-8

# In[1]:


import os
from fig_gen  import FigureGenerator as fg
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')
import missingno as msno
np.random.seed(19)
from matplotlib.pyplot import figure
data_directory = '/work2/05067/nagyz/austin_energy/data/'
filepaths = [os.path.join(data_directory,f) for f in os.listdir(data_directory) if f.endswith('.csv')]
from datetime import datetime 
from Graphs import load_data,load_graph_kwh,yearly_graph_kwh,load_graph_count,monthly_graph_kwh,daily_graph_kwh
from Graphs import load_graph_kwh_and_count,yearly_graph_kwh_and_count,monthly_graph_kwh_and_count,daily_graph_kwh_and_count
from agg_and_heatmap_functions import length,prav3,prav2,prav1,dinostorm,plot_aggraph,heatmap,finalheatmap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, SGDRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor,GradientBoostingRegressor
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
weather_daily=pd.read_csv('Austin_weather_daily.csv')
weather_daily['time']=pd.to_datetime(weather_daily['time'],utc=False)
weather_daily1=weather_daily.set_index('time',drop=True)
weather_austin_daily=weather_daily1['tavg']
# NOT UTC TIME
weather_Pred=pd.read_csv('Austin_hourly.csv')
weather_Pred['time']=pd.to_datetime(weather_Pred['time'])
weather_pred1=weather_Pred.set_index('time',drop=True)
weather_austin_hourl_pred=weather_pred1['temp']
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import SeasonalAD
from adtk.data import to_events
from adtk.detector import OutlierDetector
from scipy.stats import t
import torch
import time
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
from sklearn.metrics import roc_auc_score,recall_score
from torch.utils.data import TensorDataset
from model_fns import clean_data,make_features,preprocess_data,prediction_model,check_outlier,split_sequences
from model_fns import SelfAttention,Encoder,Decoder,RecurrentAutoencoder,train_lstmae_model,val_lstmae_model,find_threshold,predict_anomaly


# In[2]:


id_list = pd.read_csv('buildingInfo.csv')

#id_list.drop('Unnamed: 0',axis=1,inplace=True)
id_list['buildingType'].unique()
res_type = ['SINGLE FAMILY', 'MULTIFAMILY',  'FOURPLEX', 
            'DUPLEX', 'CONDOS', 'CONDO (STACKED)', 
            '1 FAM DWELLING, ACCESSORY DWELLING UNIT',
            'TOWNHOMES','MOHO SINGLE PP', 
            '1 FAM DWELLING, GARAGE APARTMENT', 'MOHO DOUBLE PP',
            'MOHO SINGLE REAL', 'MOHO DOUBLE REAL', 'TRIPLEX', 
            '1 FAM DWELLING, MOHO DOUBLE REAL',
           '1 FAM DWELLING, MOHO SINGLE REAL']
res_type_id_list = id_list[id_list['buildingType'].isin(res_type)].copy()
good_servicepoints_without_sampling=pd.read_csv('good_sp_without_sampling.csv')
good_servicepoints_with_sampling=pd.read_csv('good_sp_with_sampling.csv')


# In[3]:


single_Famiy=res_type_id_list[res_type_id_list['buildingType']=='SINGLE FAMILY']
E_Res_singlefamily=single_Famiy[single_Famiy['rate']=='E-RES']
E_Res_singlefamily.head()


# In[31]:


kmeans_lof_result_


# In[32]:


auc_baseline_lof_scores = []
for i in tqdm(E_Res_singlefamily['Service Point']):
    servicepoint=i
    print(servicepoint)
    kmeans_lof_result_,df_lstm=check_outlier(servicepoint,num_kmeans_outlier=10)
    quantile_50=kmeans_lof_result_['Energy_kwh'].quantile(0.5)
    kmeans_lof_result_[f'{servicepoint}true_anomaly']=0
    kmeans_lof_result_[f'{servicepoint}true_anomaly'] = np.where((kmeans_lof_result_['tavg'] < 5) &
                                        (kmeans_lof_result_['Energy_kwh'] < quantile_50), 1,  kmeans_lof_result_[f'{servicepoint}true_anomaly'])
    kmeans_lof_result_[f'{servicepoint}true_anomaly'] = np.where((kmeans_lof_result_['tavg'] > 30) & 
                                        (kmeans_lof_result_['Energy_kwh'] < quantile_50), 1, kmeans_lof_result_[f'{servicepoint}true_anomaly'])
    length=12
    anomalypoints=len(kmeans_lof_result_['2021-02-10':][kmeans_lof_result_['2021-02-10':][f'{servicepoint}true_anomaly']==1])
    print('number of anomaly points',anomalypoints)
    if anomalypoints>0:
        auc_baseline = roc_auc_score(kmeans_lof_result_[[f'{servicepoint}true_anomaly']], kmeans_lof_result_[[f'{servicepoint}lof_is_outlier']])
        print('auc_lof_baseline',auc_baseline)
        auc_baseline_lof_scores.append(auc_baseline)
        


# In[37]:


lof_baseline=sum(auc_baseline_lof_scores)/len(auc_baseline_lof_scores)


# In[23]:


auc_ensemble_scores = []
auc_baseline_scores = []
for i in tqdm(E_Res_singlefamily['Service Point']):
    servicepoint=i
    print(servicepoint)
    kmeans_lof_result_,df_lstm=check_outlier(servicepoint,num_kmeans_outlier=10)
    quantile_50=kmeans_lof_result_['Energy_kwh'].quantile(0.5)
    kmeans_lof_result_[f'{servicepoint}true_anomaly']=0
    kmeans_lof_result_[f'{servicepoint}true_anomaly'] = np.where((kmeans_lof_result_['tavg'] < 5) &
                                        (kmeans_lof_result_['Energy_kwh'] < quantile_50), 1,  kmeans_lof_result_[f'{servicepoint}true_anomaly'])
    kmeans_lof_result_[f'{servicepoint}true_anomaly'] = np.where((kmeans_lof_result_['tavg'] > 30) & 
                                        (kmeans_lof_result_['Energy_kwh'] < quantile_50), 1, kmeans_lof_result_[f'{servicepoint}true_anomaly'])
    length=12
    anomalypoints=len(kmeans_lof_result_['2021-02-10':][kmeans_lof_result_['2021-02-10':][f'{servicepoint}true_anomaly']==1])
    print('number of anomaly points',anomalypoints)
    if anomalypoints>0:
        dataframe=df_lstm[['difference']]
        train_sequences,val_sequences,test_sequences=split_sequences(length,dataframe)
        rnn_model = RecurrentAutoencoder(seq_len=12,n_features=1, embedding_dim=64,num_layers=2,dropout_rate=0.2)
        criterion = nn.L1Loss(reduction='sum')
        optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
        total_epochs=50
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_val_loss = float('inf')
        patience = 10
        early_stop_counter = 0
        best_model = None  # initialize variable to hold the best model

        for epoch in range(total_epochs):
          start = time.time()
          train_loss = train_lstmae_model(train_sequences, rnn_model, epoch, total_epochs, optimizer, criterion)
          val_loss = val_lstmae_model(val_sequences, rnn_model, epoch, total_epochs, criterion)
          #print(f'Epoch {epoch}-, Training_loss : {train_loss} ,Validation_loss : {val_loss}, Time :{round(start-time.time(),2)}')
          if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model = copy.deepcopy(rnn_model.state_dict())
          else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
              #print(f'Validation loss did not improve for {patience} epochs. Training stopped.')
              break
        print(best_val_loss)
        rnn_model.load_state_dict(best_model)
        th=find_threshold(rnn_model,val_sequences)
        lstm_results=predict_anomaly(rnn_model,test_sequences,th,df_lstm)
        agg_dict = {'error': 'sum', 'anomaly': lambda x: x.mode()[0]}
        final_lstm_results=lstm_results.resample('D').agg(agg_dict)
        final_lstm_results1=final_lstm_results.rename(columns={'anomaly':'lstm_anomaly'})
        final_results=final_lstm_results1.merge(kmeans_lof_result_,left_index=True,right_index=True)
        final_results[f'{servicepoint}majority_vote'] = final_results[['lstm_anomaly', 
                                                        f'{servicepoint}lof_is_outlier', f'{servicepoint}k_means_is_outlier']].mode(axis=1)[0]
        auc_ensemble = roc_auc_score(final_results[[f'{servicepoint}true_anomaly']], final_results[[f'{servicepoint}majority_vote']])
        auc_baseline = roc_auc_score(final_results[[f'{servicepoint}true_anomaly']], final_results[[f'{servicepoint}lof_is_outlier']])
        print('ensembe',auc_ensemble,'auc_baseline',auc_baseline)
        auc_ensemble_scores.append(auc_ensemble)
        auc_baseline_scores.append(auc_baseline)


# In[35]:


ensemble_baseline=sum(auc_ensemble_scores)/len(auc_ensemble_scores)


# In[34]:


kmeans_baseline=sum(auc_baseline_scores)/len(auc_baseline_scores)


# In[39]:


difference=ensemble_baseline-kmeans_baseline
100*difference/(kmeans_baseline)


# In[42]:


avg_kmeans_lof=(kmeans_baseline+lof_baseline)/2
avg_kmeans_lof


# In[45]:


difference1=ensemble_baseline-avg_kmeans_lof
100*difference1/(avg_kmeans_lof)

