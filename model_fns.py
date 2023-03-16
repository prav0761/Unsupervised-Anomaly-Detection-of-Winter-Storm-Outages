
# coding: utf-8

# In[14]:


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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset


# In[15]:


def clean_data(servicepoint):
    data=load_data(servicepoint)
    data=data.set_index(data[data.columns[0]])
    data=data[[f'{servicepoint} kWh Value']]['2018':].resample('H').sum()
    data = data.rename(columns={list(data.columns.values)[0]: 'Energy_kwh'})
    data1=data[:'2021-02-10']
    bad_value=data1[data1.columns[0]].quantile(0.98)
    #print('filter_value',bad_value)
    Index=data1[data1['Energy_kwh']>bad_value].index
    data1=data1.drop(Index)
    return pd.concat([data1,data['2021-02-11':'2021-02-22']])
def make_features(dataframe,time_level):
    s = dataframe.copy()
    if time_level!='hourly':
        example_ts = s.resample('D').mean()
    else:
        example_ts=s
    example_ts['dayofyear_cos'] = np.cos(2 * np.pi * example_ts.index.dayofyear / 365)
    example_ts['dayofyear_sim'] = np.sin(2 * np.pi * example_ts.index.dayofyear / 365)
    example_ts['month_cos'] = np.cos(2 * np.pi * example_ts.index.month / 12)
    example_ts['month_sin'] = np.sin(2 * np.pi * example_ts.index.month / 12)
    example_ts['year'] = example_ts.index.year

    if time_level == 'hourly':
        example_ts['TMP'] = weather_austin_hourl_pred.reindex(example_ts.index).values
    else:
        example_ts['TMP'] = weather_austin_daily.resample(time_level).mean().reindex(example_ts.index).values
    return example_ts


# In[16]:


def preprocess_data(servicepoint):
    s=clean_data(servicepoint)
    s = s.rename(columns={list(s.columns.values)[0]: 'Energy_kwh'})
    train_set=s['2018-01-01':'2021-02-10']
    test_set=s['2021-02-11':'2021-02-21']
    residential_train=make_features(train_set,'hourly')
    residential_test=make_features(test_set,'hourly')
    Q1=residential_train[residential_train.columns[0]].quantile(0.25)
    Q3=residential_train[residential_train.columns[0]].quantile(0.75)
    IQR=Q3-Q1
    upper_bound=Q3+1.5*IQR
    Index=residential_train[(residential_train['Energy_kwh'] <Q1 ) & (residential_train['TMP'] <0) ].index
    #print('dropped_points',len(Index))
    cleaned_train=residential_train.drop(Index,axis=0)
    cleaned_train1=pd.concat([cleaned_train,residential_test])
    return cleaned_train1


# In[17]:


def prediction_model(service_point,visualize=True,calculate_metrics=None,metric=None,boostrap=None,outlier_go=None):
    q=preprocess_data(service_point)
    if boostrap==True:
        def bootstrap_sample(data):
            return data.sample(frac=1, replace=True,random_state=567)
        temp_5=q[q['TMP']<=0][:'2020-01-01']
        n_samples = 5
        bootstrapped_data = [bootstrap_sample(temp_5) for i in range(n_samples)]
        bootstrapped_data = pd.concat(bootstrapped_data, axis=0)
        q1=pd.concat([q,bootstrapped_data])
        train_below_5=q1[q1['TMP']<=5].sort_index()[:'2020-01-01']
        train_above_5=q1[q1['TMP']>5].sort_index()[:'2020-01-01']
    else:
        train_below_5=q[q['TMP']<=5][:'2020-01-01']
        train_above_5=q[q['TMP']>5][:'2020-01-01']


    m_below,b_below = np.polyfit(train_below_5['TMP'], train_below_5['Energy_kwh'], 1)
    if visualize:

        plt.scatter(train_below_5['TMP'], train_below_5['Energy_kwh'])
        plt.plot(train_below_5['TMP'], m_below*train_below_5['TMP'] +b_below,color='red')
        plt.show()
        corr_check=train_below_5['TMP'].corr(train_below_5['Energy_kwh'])
        print(train_below_5['TMP'].corr(train_below_5['Energy_kwh']))

    m_above,m1_above,b_above = np.polyfit(train_above_5['TMP'], train_above_5['Energy_kwh'], 2)
    if visualize:
        plt.scatter(train_above_5['TMP'], train_above_5['Energy_kwh'])
        plt.plot(train_above_5['TMP'], m_above*train_above_5['TMP']**2 + m1_above*train_above_5['TMP'] + b_above,color='red')
        plt.show()
        print(train_above_5['TMP'].corr(train_above_5['Energy_kwh']))


    test_below_5=q[q['TMP']<=5]['2020-01-01':]
    test_above_5=q[q['TMP']>5]['2020-01-01':]

    #SCALING 
    scaler_below = MinMaxScaler()
    scaler_pred_below=MinMaxScaler()
    scaler_pred_below.fit(train_below_5[['Energy_kwh']])

    train_below_5_scaled = scaler_below.fit_transform(train_below_5.to_numpy())
    train_below_5_scaled = pd.DataFrame(train_below_5_scaled, columns=train_below_5.columns,index=train_below_5.index)

    test_below_5_scaled = scaler_below.transform(test_below_5.to_numpy())
    test_below_5_scaled = pd.DataFrame(test_below_5_scaled, columns=test_below_5.columns,index=test_below_5.index)

    scaler_above = MinMaxScaler()
    scaler_pred_above=MinMaxScaler()
    scaler_pred_above.fit(train_above_5[['Energy_kwh']])

    train_above_5_scaled = scaler_above.fit_transform(train_above_5.to_numpy())
    train_above_5_scaled = pd.DataFrame(train_above_5_scaled, columns=train_above_5.columns,index=train_above_5.index)

    test_above_5_scaled = scaler_above.transform(test_above_5.to_numpy())
    test_above_5_scaled = pd.DataFrame(test_above_5_scaled, columns=test_above_5.columns,index=test_above_5.index)


    ## FITTING
    lr_below_5_score_model=LinearRegression()
    Lr_below_5=LinearRegression().fit(train_below_5_scaled[['TMP']], train_below_5_scaled[['Energy_kwh']])
    final_rf=RandomForestRegressor()
    final_rf.fit(train_above_5_scaled[train_above_5_scaled.columns[1:]], train_above_5_scaled[['Energy_kwh']])
    lr_pred_below_5=Lr_below_5.predict(test_below_5_scaled[['TMP']])
    lr_pred_below_5_up=scaler_pred_below.inverse_transform(np.array(lr_pred_below_5).reshape(-1,1))

    below_5_predictions_sc=lr_pred_below_5_up


    below_5_predictions=pd.DataFrame(below_5_predictions_sc,
                                     index=test_below_5_scaled.index)


    test_below_5_scaled2=pd.DataFrame(scaler_below.inverse_transform(test_below_5_scaled),columns=test_below_5_scaled.columns,
                                      index=test_below_5_scaled.index)
    below_5_predictions1=test_below_5_scaled2.merge(below_5_predictions,how='inner',left_index=True,right_index=True)       

    test_rf_features = test_above_5_scaled[test_above_5_scaled.columns[1:]]
    pred_above_5=final_rf.predict(test_rf_features)
    pred_above_5_up=scaler_pred_above.inverse_transform(np.array(pred_above_5).reshape(-1,1))


    above_5_predictions=pd.DataFrame(pred_above_5_up,
                                     index=test_above_5_scaled.index)


    test_above_5_scaled2=pd.DataFrame(scaler_above.inverse_transform(test_above_5_scaled),columns=test_above_5_scaled.columns,
                                      index=test_above_5_scaled.index)

    above_5_predictions1=test_above_5_scaled2.merge(above_5_predictions,how='inner',left_index=True,right_index=True)



    ## JOINING EVERYTHING AND CONVERTING BACK
    total_predictions=pd.concat([above_5_predictions1,below_5_predictions1])[['Energy_kwh','TMP',0]]
    total_predictions1=total_predictions[total_predictions[0]>0].sort_index().rename(
                                                            columns={0: 'prediction'})
    total_predictions1['difference']=total_predictions1['Energy_kwh']-total_predictions1['prediction']
    total_predictions2_outlier=total_predictions1[['Energy_kwh','prediction','difference']].resample('D').sum()
    total_predictions2_true_label=total_predictions1[['Energy_kwh','prediction','difference']]
    total_predictions2_outlier['%_difference']=100*total_predictions2_outlier['difference']/total_predictions2_outlier['Energy_kwh']
    total_predictions2_true_label['%_difference']=100*total_predictions2_true_label['difference']/total_predictions2_true_label['Energy_kwh']
    return total_predictions2_outlier.merge(weather_austin_daily,left_index=True,right_index=True), total_predictions2_true_label.merge(weather_austin_hourl_pred,left_index=True,right_index=True)


# In[18]:


def check_outlier(servicepoint,num_kmeans_outlier,outlier_go=True):
    new,_=prediction_model(servicepoint,visualize=False,calculate_metrics=False,metric=None)
    new1=new[['tavg','difference']]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(new1)
    max_score = -1
    optimal_k = None
    list_k = range(2, 10)
    for k in list_k:
        kmeans = KMeans(n_clusters=k,random_state =10)
        kmeans.fit(df_scaled)
        labels = kmeans.labels_
        score = silhouette_score(df_scaled, labels)
        if score > max_score:
            max_score = score
            optimal_k = k
    labels = kmeans.labels_
    distance = kmeans.transform(df_scaled)
    outlier_indexes = np.argsort(np.min(distance, axis=1))[::-1][:num_kmeans_outlier]
    new1['k_means_is_outlier']=0
    new1.loc[new1.index.isin(new1.iloc[outlier_indexes, :].index), 'k_means_is_outlier']=1
    k_means_df=new1.merge(new[['Energy_kwh','prediction']],left_index=True,right_index=True,how='inner')

    new_diff = new[['difference']].resample('D').ffill()
    s_train = validate_series(new_diff)
    outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.01))
    anomalies = outlier_detector.fit_detect(s_train)
    outlier_df=pd.DataFrame(anomalies,columns=["lof_is_outlier"])
    outlier_df['lof_is_outlier'] = outlier_df['lof_is_outlier'].map({True: 1, False: 0})
    lof_df=new.merge(outlier_df,left_index=True,right_index=True)
    ensembe_df=lof_df.merge(k_means_df[['k_means_is_outlier']],left_index=True,right_index=True)
    ensembe_df.rename(columns={'%_difference':f'mean_{servicepoint}',
                               'k_means_is_outlier':f'{servicepoint}k_means_is_outlier',
                                'lof_is_outlier':f'{servicepoint}lof_is_outlier' },inplace=True)
    return ensembe_df[[f'mean_{servicepoint}',f'{servicepoint}k_means_is_outlier',f'{servicepoint}lof_is_outlier']].merge(
                        new[['difference','Energy_kwh','prediction','tavg']],left_index=True,right_index=True),_


# In[19]:


def split_sequences(length,dataframe):
  seq_len=length
  scaler = StandardScaler()
  X_train=dataframe[:'2021-02-10']
  length = len(X_train)
  split_index = int(length * 0.5)
  split_index = split_index - split_index % seq_len
  X_train1 = scaler.fit_transform(X_train[:split_index])
  X_val = scaler.transform(X_train[split_index:split_index + (len(X_train) - split_index) // seq_len * seq_len])
  X_test=scaler.transform(dataframe['2021-02-11 0:00:00':'2021-02-20'])
  train_sequences = np.split(X_train1, X_train1.shape[0]//seq_len)
  train_sequences = np.stack(train_sequences, axis=0)
  val_sequences = np.split(X_val, X_val.shape[0]//seq_len)
  val_sequences = np.stack(val_sequences, axis=0)
  test_sequences = np.split(X_test, X_test.shape[0]//seq_len)
  test_sequences = np.stack(test_sequences, axis=0)
  return train_sequences,val_sequences,test_sequences
# In[20]:


import torch.nn.functional as F
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        attention_weights = torch.matmul(query, key.transpose(0,1)) / torch.sqrt(torch.tensor(self.hidden_size))
        attention_weights = F.softmax(attention_weights, dim=1)
        
        attended_representation = torch.matmul(attention_weights, value)
        
        return attended_representation


# In[21]:


class Encoder(nn.Module):
  def __init__(self, n_features, embedding_dim,num_layers,dropout_rate):
    super(Encoder, self).__init__()
    self.num_layers=num_layers
    self.n_features= n_features
    self.ho_shape=embedding_dim
    self.embedding_dim=embedding_dim
    self.hidden_dim = embedding_dim
    self.lstm_encoder = nn.ModuleList()
    self.dropout_encoder = nn.ModuleList()
    self.relu=nn.ReLU()
    self.lstm_encoder.append(nn.LSTM(input_size=self.n_features,hidden_size=self.hidden_dim,batch_first=True))
    self.dropout_encoder.append(nn.Dropout(dropout_rate))
    for i in range(1,self.num_layers):
        self.hidden_dim = self.hidden_dim // 2
        self.lstm_encoder.append(nn.LSTM(input_size=self.embedding_dim,hidden_size=self.hidden_dim,batch_first=True))
        self.embedding_dim=self.hidden_dim
        if i<self.num_layers-1:
          self.dropout_encoder.append(nn.Dropout(dropout_rate))

  def forward(self, x):
    x=x.unsqueeze(0)
    h0 = torch.zeros(1, x.size(0), self.ho_shape)
    c0 = torch.zeros(1, x.size(0), self.ho_shape)
    h0 = h0.to('cuda')
    c0 = c0.to('cuda')
    x = x.to('cuda')
    for i in range(self.num_layers):
        x, (hidden_n, c0) = self.lstm_encoder[i](x,(h0,c0))
        hidden_n = hidden_n[:, :, :hidden_n.size(2) // 2]
        c0 = c0[:, :, :c0.size(2) // 2]
        h0, c0 = hidden_n, c0
        #print(x.shape,hidden_n.shape)
        x=self.relu(x)
        if i<self.num_layers-1:
          x = self.dropout_encoder[i](x)

    encoded = x.squeeze(0)
    return encoded


class Decoder(nn.Module):
  def __init__(self, seq_len,input_dim, n_features,num_layers,dropout_rate):
    super(Decoder, self).__init__()
    self.seq_len=seq_len
    self.input_dim=input_dim
    self.hidden_dim = input_dim*2
    self.num_layers=num_layers
    self.n_features = n_features
    self.lstm_decoder = nn.ModuleList()
    self.dropout_decoder = nn.ModuleList()
    self.lstm_decoder.append(nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, batch_first=True))
    self.relu=nn.ReLU()
    self.dropout_decoder.append(nn.Dropout(dropout_rate))
    for i in range(1,self.num_layers):
      self.output_hidden_dim=self.hidden_dim//2
      self.lstm_decoder.append(nn.LSTM(input_size=self.hidden_dim, hidden_size=self.output_hidden_dim, batch_first=True))
      self.hidden_dim=self.output_hidden_dim
      if i<self.num_layers-1:
          self.dropout_decoder.append(nn.Dropout(dropout_rate))
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):
    x=x.unsqueeze(0)
    h0 = torch.zeros(1, x.size(0), self.input_dim*2)
    c0 = torch.zeros(1, x.size(0), self.input_dim*2)
    h0 = h0.to('cuda')
    c0 = c0.to('cuda')
    x = x.to('cuda')
    #x=x.repeat(1,self.seq_len,1)
    for i in range(self.num_layers):
        x, (hidden_n, _) = self.lstm_decoder[i](x,(h0,c0))
        hidden_n = hidden_n[:, :, :hidden_n.size(2) // 2]
        c0 = c0[:, :, :c0.size(2) // 2]
        h0, c0 = hidden_n, c0
        x=self.relu(x)
        if i<self.num_layers-1:
          x = self.dropout_decoder[i](x)
    return self.output_layer(x)



class RecurrentAutoencoder(nn.Module):
  def __init__(self,seq_len, n_features, embedding_dim,num_layers,dropout_rate):
    super(RecurrentAutoencoder, self).__init__()
    self.encoder = Encoder( n_features,embedding_dim,num_layers,dropout_rate)
    self.output_encoder_shape=embedding_dim//(2**(num_layers-1))
    self.attention=SelfAttention(self.output_encoder_shape)
    self.decoder = Decoder(seq_len,self.output_encoder_shape, n_features,num_layers,dropout_rate)
  def forward(self, x):
    x1 = x.to('cuda')
    x = self.encoder(x)
    x= self.attention(x)
    x = self.decoder(x)
    return x


# In[22]:


rnn_model = RecurrentAutoencoder(seq_len=12,n_features=1, embedding_dim=64,num_layers=2,dropout_rate=0.2)
criterion = nn.L1Loss(reduction='sum')
train_g=[]
val_g=[]
optimizer = optim.Adam(rnn_model.parameters(), lr=0.005)
total_epochs=100
def train_lstmae_model(data,model,epoch,total_epochs,optimizer,criterion):
    model = model.to('cuda')
    model.train()
    trainloss = torch.tensor(0.0).to('cuda')
    len = data.shape[0]
    for batch in data:
        x1 = torch.from_numpy(batch).float().to('cuda')
        optimizer.zero_grad()
        outputs = model(x1)
        loss = criterion(outputs, x1)
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
    trainloss = trainloss.cpu().item()
    return round(trainloss / len, 4)
def val_lstmae_model(data,model,epoch,total_epochs,criterion):
    model.eval()
    valloss=0.0
    len=data.shape[0]
    with torch.no_grad():
      for batch in data:
        x1 = torch.from_numpy(batch).float().to('cuda')
        outputs = model(x1)
        loss = criterion(outputs, x1)
        valloss+=loss.item()
    return round(valloss/len,4)
def find_threshold(model, data):
    model = model.to('cuda')
    criterion = nn.L1Loss(reduction='sum')
    errors = []
    model.eval()
    len=data.shape[0]
    with torch.no_grad():
      for batch in data:
        x1 = torch.from_numpy(batch).float().to('cuda')
        outputs = model(x1)
        loss = criterion(outputs, x1)
        errors.append(loss.item())
    return np.quantile(errors, 0.95)


# In[23]:


def predict_anomaly(model, data,threshold,index_df):
    criterion = nn.L1Loss(reduction='sum')
    errors = []
    anomalies =[]
    model.eval()
    len=data.shape[0]
    with torch.no_grad():
      for batch in data:
        x1 = torch.from_numpy(batch).float().to('cuda')
        outputs = model(x1)
        loss = criterion(outputs, x1)
        errors.append(loss.item())
        if loss > threshold:
            anomalies.append(1)
        else:
            anomalies.append(0)
    df = pd.DataFrame({'error': errors, 'anomaly': anomalies},index=index_df['2021-02-11 0:00:00':'2021-02-20'].resample('12H').sum().index)
    #agg_dict = {'error': 'sum', 'anomaly': lambda x: x.mode()[0]}
    return df

