# 필독
# GTM 기반 특허맵을 plot하기 위해서는 서브클래스 수를 한정해야함(서브클래스 = IPC코드 종류)
# 너무 수가 적은 class들은 지워야하는데 자동화 도전

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

from collections import Counter

# drive.mount('/content/drive') #구글 드라이브 접속하려면 이거 써야함
# %cd ./drive/Shareddrives/AI_Rights/

### Data 불러오기

df = pd.read_csv('./data/Battery.xlsx', encoding='CP949')

df.head()

print(len(df)) # 총 데이터 개수

for i in df.columns: #column 명
  print(i)

df = df.drop(['공개번호', '공개일', '우선권 번호', '우선권 국가', '우선권 주장일', '국제 출원일'], axis = 1)

KR_df = df[df['국가코드'] == 'KR'] #한국 데이터만 추출
KR_df = KR_df.reset_index(drop=True)

df = df[df['국가코드'] != 'KR'] #비 한국은 중국, 미국, 일본 3개국가만 존재(중국 15999, 미국 1101, 일본 1547개)
df = df.reset_index(drop=True)

print(len(df))
print(len(KR_df))

Counter(df['국가코드'])

df.info() #Dtype 확인

## 코드 분석

df['Original IPC All']

IPC_list = []
for i in df["Original IPC All"]:
  lst = str(i).split('|')
  for j in lst:
    j=j.strip()
    IPC_list.append(j[:4])
IPC_list

#여기부터 수정
IPC_Count = Counter(IPC_list)
len(IPC_Count)

#정성적으로 IPC 코드 수 보고 적당이 cut_line지정, 200개 > 30개 수준 생각
cut_line = 50
sub_class = [k for k, v in IPC_Count.items() if v >= cut_line]
SET = set(sub_class)

print(len(SET))

set_list = []
for i in range(len(df)):
  in_lst = set()
  lst = str(df.iloc[i]["Original IPC All"]).split('|')
  for j in lst:
    j = j.strip()
    in_lst.add(j[:4])
  set_list.append(list(in_lst))
print(set_list)

SET = sorted(list(SET))

table = pd.DataFrame(index = df["Original IPC All"].index, columns = SET)
table

for i in range(len(df)):
  for j in SET:
    if j in set_list[i]:
      table.iloc[i][j] = 1
table

#서브클래스에 해당 안하는 극소수의 데이터들 삭제
table = table.fillna(0)
del_list = []
for i in range(len(table)-1,0, -1):
  if sum(table.iloc[i]) == 0:
    del_list.append(i)
    table = table.drop(([table.index[i]]))
len(table)

table

for i in del_list:
  set_list.pop(i)

#table = table.drop(['nan'], axis = 1)
print(table.shape)

# 한국 특허 따로 분류

kr_list = []
for i in range(len(KR_df)):
  in_lst = set()
  lst = str(KR_df.iloc[i]["Original IPC All"]).split('|')
  for j in lst:
    j = j.strip()
    in_lst.add(j[:4])
  kr_list.append(list(in_lst))
print(kr_list)

len(KR_df)

kr_table = pd.DataFrame(index = KR_df["Original IPC All"].index, columns = SET)
kr_table

for i in range(len(KR_df)):
  for j in SET:
    if j in kr_list[i]:
      kr_table.iloc[i][j] = 1
kr_table

#kr_table = kr_table.drop(['nan'], axis = 1)
kr_table.info()

kr_table

#서브클래스에 해당 안하는 극소수의 데이터들 삭제
kr_table = kr_table.fillna(0)
kr_del_list = []
for i in range(len(kr_table)-1,0, -1):
  if sum(kr_table.iloc[i]) == 0:
    kr_del_list.append(i)
    kr_table = kr_table.drop(([kr_table.index[i]]))
len(kr_table)

for i in kr_del_list:
  kr_list.pop(i)

#kr_table = kr_table.drop(['nan'], axis = 1)

print(kr_table.shape)

code = [1624, 11463, 4347, 401, 19, 207, 358]

for i in code:
  print(f'{i} index에 해당하는 IPC는 아래와 같다')
  print(df['Original IPC All'][i])
  print('\n')

#GTM 도전

# !pip install ugtm

'''
from ugtm import eGTM
model = eGTM(k=5, m=3)
fit_model = model.fit(table)
vec_table = fit_model.transform(table)

vec_table

x = vec_table[:,0]
y = vec_table[:,1]
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('vec_table')
plt.scatter(x, y)
plt.show()
'''

import ugtm

gtm = ugtm.runGTM(table, k=6, m = 3)
#새로 들어올 데이터에 따라 k, m, s, reg 설정해야함
# k, m, s, reg의 경우 gtm을 구성할 때 사용하는 rbf 함수의 parameter인데 설명할 때는 clustering이 잘 이루어지는 parameter를 정성적으로 grid search해서 가장 적절히 분리되는 parameter를 찾았다고 설명

gtm_modes = gtm.matModes

# The four GTM hyperparameters (regul, k, m, and s) can be
# tuned: k is used for tuning the GTM resolution (a GTM
# map is discretized into a grid of [k, k] nodes), m is the
# number of RBF functions (defining an [m, m] grid), s is the
# RBF function width factor, and regul is the regularization
# coefficient. Implementation details can be found in the
# API description.

dgtm_modes_k9_T600_new = pd.DataFrame(gtm_modes, columns=["x1", "x2"])
#dgtm_modes["label"] = y


ddf = dgtm_modes_k9_T600_new
print(ddf)
ddf.to_csv('GTM (V2_ics_robot_new.csv')

index = []
grid = [-1, -3/5, -1/5, 1/5, 3/5, 1]
for i in grid:
  for j in grid:
    bd = ddf[round(ddf['x1'],2) == i]
    bd = bd[round(bd['x2'],2) == j]
    index.append(bd.index)

for i in range(0,36):
  for j in index[i]:
    set_list[j]

last = []
for k in range(0,36):
  aa = set()
  for i in index[k]:
    for j in (set_list[i]):
      aa.add(j)
  #aa = list(aa)
  last.append(aa)
  print(aa)

row_a = last[0] | last[6] | last[12] | last[18] | last[24] | last[30]
row_b = last[1] | last[7] | last[13] | last[19] | last[25] | last[31]
row_c = last[2] | last[8] | last[14] | last[20] | last[26] | last[32]
row_d = last[3] | last[9] | last[15] | last[21] | last[27] | last[33]
row_e = last[4] | last[10] | last[16] | last[22] | last[28] | last[34]
row_f = last[5] | last[11] | last[17] | last[23] | last[29] | last[35]

row = []
row.append(row_a)
row.append(row_b)
row.append(row_c)
row.append(row_d)
row.append(row_e)
row.append(row_f)

col_a = last[0] | last[1] | last[2] | last[3] | last[4] | last[5]
col_b = last[6] | last[7] | last[8] | last[9] | last[10] | last[11]
col_c = last[12] | last[13] | last[14] | last[15] | last[16] | last[17]
col_d = last[18] | last[19] | last[20] | last[21] | last[22] | last[23]
col_e = last[24] | last[25] | last[26] | last[27] | last[28] | last[29]
col_f = last[30] | last[31] | last[32] | last[33] | last[34] | last[35]

col = []
col.append(col_a)
col.append(col_b)
col.append(col_c)
col.append(col_d)
col.append(col_e)
col.append(col_f)

llast = []
for i in range(6):
  for j in range(6):
    llast.append(col[i] & row[j])

asdf=[]
for i in range(0,36):
  asdf.append(i)

b = [0,6,12,18,24,30,7,8,9,10,11]
aaasdf = [x for x in asdf if x not in b]

aaasdf

from joblib.parallel import eval_expr
ab = row_a & col_b
ac = row_a & col_c
bc = row_b & col_c
de = row_d & col_e
df = row_d & col_f

ae = row_a & row_e

db = row_d & col_b
fb = row_f & col_b
ec = row_e & col_c
fd = row_f & col_d
ee = row_e & col_e
ef = row_e & col_f
ff = row_f & col_f

#-ab-ac-bc-de-df-db-fb-ec-fd-ee-ef-ff

ae

print('총체적인 공백기술')
print(f'1번 공백기술은ab {ab-ac-bc-de-df-db-fb-ec-fd-ee-ef-ff}')
print(f'2번 공백기술은ac {ac-ab-bc-de-df-db-fb-ec-fd-ee-ef-ff}')
print(f'3번 공백기술은bc {bc-ab-ac-de-df-db-fb-ec-fd-ee-ef-ff-col_d}')#데이터가 너무 많이 존재해 더 많은 데이터의 차집합을 구함
print(f'4번 공백기술은de {de-ab-ac-bc-df-db-fb-ec-fd-ee-ef-ff-row_c}')#데이터가 너무 많이 존재해 더 많은 데이터의 차집합을 구함
print(f'5번 공백기술은df {df-ab-ac-bc-de-db-fb-ec-fd-ee-ef-ff-row_b}')#데이터가 너무 많이 존재해 더 많은 데이터의 차집합을 구함

print('\n')
print('국내  부족기술')
print(f'1번 부족기술은db {db-ab-ac-bc-de-df-fb-ec-fd-ee-ef-ff}')
print(f'2번 부족기술은fb {fb-ac-de-ec-fd-ee}')#아예 데이터가 존재하지 않아 인접데이터 차용
print(f'3번 부족기술은ec {ec-ab-ac-bc-de-df-db-fb-fd-ee-ef-ff}')
print(f'4번 부족기술은fd {fd-ab-ac-bc-de-df-db-fb-ec-ee-ef-ff-row_c-row_b}')
print(f'5번 부족기술은ee {ee-ab-ac-bc-de-df-db-fb-ec-fd-ef-ff-row_c-row_b}')
print(f'6번 부족기술은ef {ef-ab-ac-bc-de-df-db-fb-ec-fd-ee}')#아예 데이터가 존재하지 않아 인접데이터 차용
print(f'7번 부족기술은ff {ff-ab-ac-bc-de-df-db-fb-ec-fd-ee-ef-col_d-row_c-row_b}')
print(f'8번 부족기술은ae {ae-ff-bc-df-db-fb-fd-col_d}')

gtm_modes

grid = [-1, -3/5, -1/5, 1/5, 3/5, 1]
import matplotlib.pyplot as plt
x = gtm_modes[:,0]
y = gtm_modes[:,1]
plt.figure(figsize = [15,15])
plt.grid(False)
for i in grid:
  plt.axhline(i, color='black')
  plt.axvline(i, color='black')
plt.scatter(x,y, s=1000, c = 'blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('vec_table')
#plt.scatter(x, y)
#plt.show()
plt.savefig('world_gtm.png')

# ex. with discrete labels and inter-node interpolation
gtm.plot_multipanel(output="sub_ver_k5",labels=set_list,discrete=True,pointsize=20)

transformed=ugtm.transform(optimizedModel=gtm,train = table,test=kr_table)

trans_modes = transformed.matModes

trans_modes

grid = [-1, -3/5, -1/5, 1/5, 3/5, 1]
x = trans_modes[:,0]
y = trans_modes[:,1]
plt.figure(figsize = [15,15])
plt.grid(False)
for i in grid:
  plt.axhline(i, color='black')
  plt.axvline(i, color='black')
plt.scatter(x,y, s=1000, c = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('vec_table')
#plt.scatter(x, y)
#plt.show()
plt.savefig('korea_gtm.png')

print(table.shape)
print(kr_table.shape)

#해외, 국내 나눌 때 나라별로 색깔 통일해서 찍을 예정 코드

#run model on train
gtm = ugtm.runGTM(table,doPCA=True)

#test new data (test)
transformed=ugtm.transform(optimizedModel=gtm,train=table,test=kr_table,doPCA=True)

transformed.plot_multipanel(output="test_go",labels=set_list,discrete=True,pointsize=20)

#plot transformed test (html)
transformed.plot_html(output="testout14",pointsize=20)

#plot transformed test (pdf)
transformed.plot(output="testout15",pointsize=20)

#plot transformed data on existing classification model,
#using training set labels
gtm.plot_html_projection(output="testout16",projections=transformed, discrete=True,pointsize=20)

