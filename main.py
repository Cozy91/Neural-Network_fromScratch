import numpy as np 
import pandas as pd
import random 

df=pd.read_csv("~/projects/ml_project/games_data_goty.csv")

df['r-date']=pd.to_datetime(df['r-date'])
df['Goty']=df['Goty'].fillna(0).astype(int)
df['year']=df['r-date'].dt.year

df['user score']=pd.to_numeric(df['user score'],errors='coerce')
df['user score']=df['user score'].fillna(df['user score'].median())

features=['score','user score','critics','users']

train_df=df[df['year']<2014].copy()
test_df=df[df['year'] >= 2014].copy()

X_train=train_df[features].values.astype(float)
y_train=train_df['Goty'].values.astype(float)

X_test=test_df[features].values.astype(int)
y_test=test_df['Goty'].values.astype(float)
#normalisation as score(0-100) and users 0-100000(idk)

mean=X_train.mean(axis=0)
std=X_train.std(axis=0)+ 1e-8
X_train=(X_train-mean)/std
X_test=(X_test-mean)/std #we will divide it from the same mean cuz if we divide it by the mean of X_test, it would be unfair and not possible irl cuz that is future data  
                        #we also assume that the test data will be similar to the training data so we are using the same mean and std to normalise it 

random.seed(60)

input=len(features) # 4 features here
h1=16
output=1 

input_weights=random.randn(input,h1) * 0.1 #4*16 matrix of random weights 
b1_h1=random.randn(1,h1) * 0.1 #16 random biases for 1st hidden layer 
h1_weights=np.random.randn(h1,1) * 0.1 #h1-output
b2_weights=random.randn(1,output) * 0.1 #8 random biases for h2 

def ReLU(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward_prop(input_weights,b1_h1,X_train,b2_weights):
    z1= X_train.dot(input_weights)+b1_h1 #this is also a matrix, take row from X_train and multiply that to column of input_weights

    a1=ReLU(z1)
    z2=a1.dot(h1_weights) + b2_weights
    a2=sigmoid(z2) #final output 

    return z1,a1,z2,a2 

def bce(y_pred,y_actual):
    k=1e-8
    return -np.mean(y_actual*np.log(y_pred+k) + (1-y_actual)*np.log(1-y_pred+k))

pos_weight=(y_train==0).sum()/max((y_train==1).sum(),1) # will use when data is imbalanced which it is lol idk really how to use this tbh i just know it gives equal priority to boht the options in the data so that the data is not biased anymore 

epocs=100 

def backprop()
