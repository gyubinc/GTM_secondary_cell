# 한번 모두실행 하고 기다렸다가 런타임 다시시작 및 다시실행 누를 것

# EDA

### 드라이브 환경 세팅 및 import

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# %matplotlib inline

# from google.colab import drive

import matplotlib.ticker as ticker
from matplotlib import dates
import datetime as dt
from collections import Counter as C

# !pip install darts
# !pip install utils

# drive.mount('/content/drive') #구글 드라이브 접속하려면 이거 써야함
# %cd ./drive/Shareddrives/AI_Rights/

### Data 불러오기

df = pd.read_csv('./data/Battery.xlsx', encoding='CP949') #데이터 불러오기

df.head(1)

df['국가코드'].unique()

KR = df[df['국가코드']=='KR']
CN = df[df['국가코드']=='CN']
US = df[df['국가코드']=='US']
JP = df[df['국가코드']=='JP']

print(len(df)) # 총 데이터 개수

for i in df.columns: #column 명
  print(i)

df2 = df.copy()
df2['출원일'] = df2['출원일'].apply(lambda x: str(dt.datetime.strptime(x[:7], "%Y-%m"))[:7])
sorted_df = df.sort_values("출원일")

df.info() #Dtype 확인

## 월 단위 개수 plotting

from scipy.ndimage.filters import gaussian_filter1d
def smoothing(y):
  ysmoothed = gaussian_filter1d(y, sigma=2)
  return ysmoothed

All = df.copy()
month_18 = ['2021-05','2021-05']
def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

def count(df2):
  df2['출원일'] = df2['출원일'].apply(lambda x: str(dt.datetime.strptime(x[:7], "%Y-%m"))[:7]) #월 단위 자름
  sorted_df = df2.sort_values('출원일')
  Count = C(sorted_df['출원일'])
  return Count

def month_plotting(ctry,xline, *df2):
  plt.figure(figsize=(20,8))
  ax=plt.axes()
  ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
  for i in (df2):
    name = get_df_name(i)
    Count = count(i)
    plt.plot(list(Count.keys()),list(Count.values()),label=name)
  plt.plot(xline,[0,500],label='2021-05')
  plt.legend()
  plt.xticks(rotation=90,)
  plt.xlabel('Month')
  plt.title(f"Applications per Month : {ctry}")
  plt.show()

month_plotting("all",month_18,All,KR,JP,US,CN,)

def month_plotting_2(ctry,xline,all, *df2):
  plt.figure(figsize=(20,8))
  ax=plt.axes()
  ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
  x_label = list(count(all).keys())
  plt.plot(x_label,[0 for i in range(len(x_label))],c='white')
  for i in (df2):
    name = get_df_name(i)
    Count = count(i)
    plt.plot(list(Count.keys()),list(Count.values()),label=name)
  plt.plot(xline,[0,35],label='2021-05')
  plt.legend()
  plt.xticks(rotation=90,)
  plt.xlabel('Month')
  plt.title(f"Applications per Month : {ctry}")
  plt.ylim(0,35)
  plt.show()

month_plotting_2("all",month_18,All,US,JP,KR,)

## 연단위 개수 plotting 및 CDF

df3 = df.copy()

df3['출원일'] = df3['출원일'].apply(lambda x: str(dt.datetime.strptime(x[:4], "%Y"))[:4]) #연 단위 자름
sorted_df_2 = df3.sort_values('출원일')
plt.figure(figsize=(16,8))
ax=plt.axes()
sns.countplot(x=sorted_df_2['출원일'],)
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.title("Applications per Year")
plt.show()

# CDF

slist = sorted_df_2['출원일'].tolist()
def to_sequential_data(slist):
  count = C(slist)
  date = count.keys()
  s_data = pd.DataFrame(count.values(),date,)
  s_data.columns=['date']
  s_data = s_data.sort_index()
  min_year = int(s_data.index[0])
  max_year = int(s_data.index[-1])
  print(min_year, max_year)
  print(max_year-min_year)
  return s_data, min_year, max_year

s_data, min_year, max_year = to_sequential_data(slist)

x=s_data.index
y=s_data['date']
cdf=np.cumsum(y)
plt.xticks(rotation=45)
plt.plot(x,cdf,marker="o",label="CDF")
plt.xlabel("X")
plt.ylabel("num of Applications")
plt.title("CDF per year")
plt.legend()
plt.show()

## 월단위 시계열 분석

### 월단위로 전처리 part

from darts.utils.missing_values import fill_missing_values
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.models import ARIMA
from darts.metrics import mape

#오류나면 runtime 재시작, 다시 실행 할 것

def return_sorted(data):
  df2 = data.copy()
  df2['출원일'] = df2['출원일'].apply(lambda x: str(dt.datetime.strptime(x[:7], "%Y-%m"))[:7])
  sorted_df = df2.sort_values("출원일")
  return sorted_df

sorted_df = return_sorted(df)

def preprocessing(data):
  slist = data['출원일'].tolist()
  count = C(slist)
  date = count.keys()
  s_data = pd.DataFrame(count.values(),date,)
  s_data.columns=['num']
  s_data['date'] = s_data.index
  s_2 = s_data
  s_data['date'] = s_data['date'].apply(lambda x: (int(x[:4])-min_year)*12 + int(x[5:7])-12)
  return s_data

s_data = preprocessing(sorted_df)

def to_timeSeries(s_data):
  s_data.index = s_data['date']
  s_data.drop(columns = ['date'],axis =1, inplace = True)
  time = {}
  s_data = s_data.sort_index()
  print(len(s_data))
  for i in range(sorted(s_data.index)[-1]):
    if i not in s_data.index:
      time[i] = 0
  v_data = pd.DataFrame(index=time.keys(),columns=['num'],data=time.values())
  s_data = s_data.append(v_data,)
  s_data['date'] = s_data.index
  min_val = min(s_data['num'])
  max_val = max(s_data['num'])
  series = TimeSeries.from_dataframe(s_data, 'date','num')
  return series, max_val, min_val

def forward(df):
  sorted_df = return_sorted(df)
  s_data = preprocessing(sorted_df)
  series, max_val, min_val = to_timeSeries(s_data)
  return series, max_val, min_val

series, max_val, min_val = forward(df)

"""
series : series 데이터 넣기
num: 에는 몇 시점 예측할 건지 기입
  ex 현재 11월 기준, num = 5시, 11, 10, 9, 8, 7 월은 이전 데이터로 예측 진행
model: 사용할 모델 함수로 입력
  from darts.models import ExponentialSmoothing
  model = ExponentialSmoothing()
"""
def model_run(series, max_val,min_val, num, model):
  train, val = series[:-num], series[-num:]
  model = model
  model.fit(train)
  plt.figure(figsize=(16,8))
  prediction = model.predict(len(val), num_samples=1000)
  series[:-num].plot(label = "actual_before 2021,05", c='black')
  series[-num:].plot(label = "actual_after 2021,05", c='red')
  x1 = x2 = len(series)-num
  y1, y2 = 0, max_val
  plt.axline((x1, y1), (x2, y2),c="green",label='18_months')
  prediction.plot(label="predicted", low_quantile=0.05, high_quantile=0.95,c='blue')
  plt.xlabel('Month')
  plt.ylabel("num of applications")
  plt.legend()
  plt.title('prediction_plot')
  plt.show()

model_run(series,max_val,min_val, 18, ExponentialSmoothing()) #Exponential

model_run(series,max_val, min_val, 18, ARIMA()) #ARIMA

##한국

series_KR,max_val,min_val = forward(KR)
model_run(series_KR, max_val, min_val, 18, ARIMA()) #ARIMA

## 중국

series_CN,max_val,min_val = forward(CN)
model_run(series_CN, max_val, min_val, 18, ARIMA()) #ARIMA

series_US,max_val,min_val = forward(US)
model_run(series_US, max_val, min_val, 18, ARIMA()) #ARIMA

series_JP,max_val,min_val = forward(JP)
model_run(series_JP, max_val, min_val, 18, ARIMA()) #ARIMA

## TFT 트랜스포머기반 시계열 예측모델

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)

date = df['출원일'].apply(lambda x: dt.datetime.strptime(x[:7],"%Y-%m"))
min(date)
date.sort_values()

s_data.sort_values(by='date')

def transform(num):
  year = 1998
  month = 12
  for i in range(num):
    month+=1
    if month==13:
      month=1
      year+=1
  year = str(year)
  month = str(month)
  return year+'-'+month+'-01'

for i in range(24):
  print(transform(i))

a = s_data.sort_values(by='date',)
date_list = []
for i in a['date']:
  date_list.append(transform(i))
date_list

s_data = s_data.sort_values('date')
s_data['date'] = date_list
s_data

series = TimeSeries.from_dataframe(s_data, 'date','num')

series = series.astype(np.float32)
training_cutoff = pd.Timestamp("20200501")
train, val = series.split_after(training_cutoff)
transformer = Scaler()
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)
series_transformed = transformer.transform(series)

# create year, month and integer index covariate series
covariates = datetime_attribute_timeseries(series, attribute="year", one_hot=False)
covariates = covariates.stack(
    datetime_attribute_timeseries(series, attribute="month", one_hot=False)
)
covariates = covariates.stack(
    TimeSeries.from_times_and_values(
        times=series.time_index,
        values=np.arange(len(series)),
        columns=["linear_increase"],
    )
)
covariates = covariates.astype(np.float32)

# transform covariates (note: we fit the transformer on train split and can then transform the entire covariates series)
scaler_covs = Scaler()
cov_train, cov_val = covariates.split_after(training_cutoff)
scaler_covs.fit(cov_train)
covariates_transformed = scaler_covs.transform(covariates)

# default quantiles for QuantileRegression
quantiles = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]
input_chunk_length = 24
forecast_horizon = 12
my_model = TFTModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=forecast_horizon,
    hidden_size=64,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=16,
    n_epochs=300,
    add_relative_index=False,
    add_encoders=None,
    likelihood=QuantileRegression(
        quantiles=quantiles
    ),  # QuantileRegression is set per default
    # loss_fn=MSELoss(),
    random_state=42,
)

my_model.fit(train_transformed, future_covariates=covariates_transformed, verbose=True)

def eval_model(model, n, actual_series, val_series):
    pred_series = model.predict(n=n, num_samples=1000)

    # plot actual series
    plt.figure(figsize=(20,8))
    actual_series[: pred_series.end_time()].plot(label="actual")

    # plot prediction with quantile ranges
    pred_series.plot(
        low_quantile=min(quantiles), high_quantile=max(quantiles), label='label_q_outer'
    )
    pred_series.plot(low_quantile=min(quantiles), high_quantile=max(quantiles), label='label_q_inner')

    plt.title("MAPE: {:.2f}%".format(mape(val_series, pred_series)))
    plt.legend()


eval_model(my_model, 18, series_transformed, val_transformed)

