
#!/usr/bin/env python
# coding: utf-8

# In[73]:


import random
import pandas as pd
import numpy as np
import sys, os
import librosa
from tqdm.auto import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings(action='ignore')
print("경로: ",os.getcwd())



#머신러닝 모델링
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from scipy.stats import uniform
import numpy as np

#import xgboost


# In[ ]:





# In[74]:


CFG = {
  "SR":16000,
  "N_MFCC":32,
  "SEED":41
}


# Fixed Random-Seed

# In[75]:


def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] =str(seed)
  np.random.seed(seed)

seed_everything(CFG['SEED'])


# Data Pre-Processing 1

# In[76]:


train_df = pd.read_csv('flask_app/Data/train_data.csv')
test_df = pd.read_csv("flask_app/Data/test_data.csv")


# In[77]:


def get_mfcc_feature(df,data_type, save_path):
  # Data Folder path
  root_folder = 'flask_app/Data/wav_dataset'
  if os.path.exists(save_path):
    print(f'{save_path}is exist.')
    return
  features = []
  for uid in tqdm(df['id']):
    root_path = os.path.join(root_folder, data_type)
    path = os.path.join(root_path, str(uid).zfill(5)+'.wav')

    # librosa패키지를 사용하여 wav 파일 load
    y, sr = librosa.load(path, sr=CFG["SR"])

    # librosa패키지를 사용하여 mfcc 추출
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG['N_MFCC'])

    y_feature = []
    # 추출된 MFCC들의 평균을 Feature로 사용
    for e in mfcc:
      y_feature.append(np.mean(e))
    features.append(y_feature)

  #기존의 자가진단 정보를 담은 데이터프레임에 추출된 오디오 Feature를 추가
  mfcc_df = pd.DataFrame(features, columns=['mfcc_'+str(x) for x in range(1,CFG["N_MFCC"]+1)])
  df = pd.concat([df, mfcc_df], axis=1)
  df.to_csv(save_path, index=False)
  print("Done")


# In[78]:


#get_mfcc_feature(train_df, 'train', "Data/train_mfcc_data.csv")
#get_mfcc_feature(test_df,'test','Data/test_mfcc_data.csv')


# In[79]:


# wav 파일의 MFCC Feature와 상태정보를 합친 학습데이터를 불러옵니다.
train_df = pd.read_csv("flask_app/Data/train_mfcc_data.csv")

# 학습데이터를 모델의 input으로 들어갈 x와 label로 사용할 y로 분할
train_x = train_df.drop(columns=['id', 'covid19'])
train_y = train_df['covid19']


# In[80]:


def onehot_encoding(ohe, x):
  # 학습데이터로부터 fit된 one-hot encoder (ohe)를 받아 transform 시켜주는 함수
  encoded = ohe.transform(x['gender'].values.reshape(-1,1))
  encoded_df = pd.DataFrame(encoded, columns=ohe.categories_[0])
  x = pd.concat([x.drop(columns=['gender']), encoded_df], axis=1)
  return x


# In[81]:


ohe = OneHotEncoder(sparse=False)
ohe.fit(train_x['gender'].values.reshape(-1,1))
train_x = onehot_encoding(ohe,train_x)


# Modeling-2

# In[82]:


model = MLPClassifier(random_state=CFG['SEED']) # Sklearn에서 제공하는 Multi-layer Perceptron classifier 사용
model.fit(train_x, train_y) # Model Train


# In[83]:


# 새로운 데이터 처리

def new_get_mfcc(path,save_path):
  #기존 파일 있을 경우
  """
  if os.path.exists(save_path):
    print(f'{save_path}is exist.')
    return
  """

  # 기침 이외의 정보
  df = pd.read_csv("flask_app/Data/new_data.csv")
  df = df.fillna(0)

  # librosa패키지를 사용하여 wav 파일 load
  y, sr = librosa.load(path, sr=CFG["SR"])

  # librosa패키지를 사용하여 mfcc 추출
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG['N_MFCC'])

  
  features = []
  y_feature = []
  # 추출된 MFCC들의 평균을 Feature로 사용
  for e in mfcc:
    y_feature.append(np.mean(e))
  features.append(y_feature)

  #기존의 자가진단 정보를 담은 데이터프레임에 추출된 오디오 Feature를 추가
  mfcc_df = pd.DataFrame(features, columns=['mfcc_'+str(x) for x in range(1,CFG["N_MFCC"]+1)])
  df = pd.concat([df, mfcc_df], axis=1)
  df.to_csv(save_path, index=False)
  print("New data Done")
  


# In[84]:


new_get_mfcc("flask_app/Data/new_cough.wav",'flask_app/Data/new_test.csv')


# In[85]:


# 위의 학습데이터를 전처리한 과정과 동일하게 test data에도 적용
test_x = pd.read_csv('flask_app/Data/new_test.csv')
test_x = test_x.drop(columns=['id'])
# Data Leakage에 유의하여 train data로만 학습된 ohe를 사용
test_x_ohe = onehot_encoding(ohe, test_x)

# Model 추론
preds = model.predict(test_x_ohe)
test_x["covid19"]=preds[0]
test_x.to_csv('flask_app/Data/new_result.csv')
print('검사결과: ',preds[0])


