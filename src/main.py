import numpy as np 
import pandas as pd
import random 

df=pd.read_csv("~/projects/ml_project/data/games_data_goty_fixed.csv")

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

input_weights=np.random.randn(input,h1) * 0.1 #4*16 matrix of random weights 
b1_h1=np.random.randn(1,h1) * 0.1 #16 random biases for 1st hidden layer 
h1_weights=np.random.randn(h1,1) * 0.1 #h1-output
b2_weights=np.random.randn(1,output) * 0.1 #8 random biases for h2 

def ReLU(x):
    return np.maximum(0,x)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))

def forward_prop(W1,b1,W2,x,b2): # W1=input_weights, b1=b1_h1,W2=h1_weights, b2=b2_weights
    z1= x.dot(W1)+b1 #this is also a matrix, take row from X_train and multiply that to column of input_weights
    a1=ReLU(z1)

    z2=a1.dot(W2) + b2 
    a2=sigmoid(z2) #final output 

    return z1,a1,z2,a2 

pos_weight=np.sqrt((y_train==0).sum()/max((y_train==1).sum(),1)) # will use when data is imbalanced which it is lol idk really how to use this tbh i just know it gives equal priority to boht the options in the data so that the data is not biased anymore 


def bce(y_pred,y): #y is true value/actual value 
    k=1e-8
    #np.clip(x, min, max)
    y_pred=np.clip(y_pred,1e-8, 1 - 1e-8)
    return -np.mean(pos_weight * y *np.log(y_pred+k) + (1-y)*np.log(1-y_pred+k))

def backprop(y_pred,y,z1,a1,x,z2,lr): #all this math took me over a day 
    global input_weights, h1_weights, b1_h1, b2_weights

    dz2=y_pred-y
    dz2[y==1] *= pos_weight
    dz2 /= len(y)
    dW2= a1.T.dot(dz2) #a1 will changer every iteration, so will the value of every other parametre except the y_actual, W2 is h1_weights  
    db2=np.sum(dz2,axis=0,keepdims=True) 

    dz1=dz2.dot(h1_weights.T) * (z1>0)
    dw1=x.T.dot(dz1) #this is input weights 
    db1=np.sum(dz1,axis=0,keepdims=True) #FINALLY DONE AFTER LIKE 6 HOURS I LOST MY SHIT DOING ALL THE MATHS

#updating weights yay
    dw1=np.clip(dw1,-1,1)
    dW2=np.clip(dW2,-1,1) #clipping so that gradient doesnt explode(it was giving NnN values lol)

    h1_weights -= lr*dW2
    b2_weights -= lr*db2 
    input_weights -= lr*dw1 
    b1_h1 -= lr*db1 


epochs = 1000  

for epoch in range(epochs):
    z1, a1, z2, a2 = forward_prop(input_weights, b1_h1, h1_weights, X_train, b2_weights)
    backprop(a2, y_train.reshape(-1,1), z1, a1, X_train, z2, 0.01)
    
    if epoch % 10 == 0:
        loss = bce(a2, y_train.reshape(-1,1))
        print(f"epoch {epoch}, Loss: {loss:.4f}")
   
print("TESTING THE DATA NOW:")
_, _, _, a2 = forward_prop(input_weights, b1_h1, h1_weights, X_test, b2_weights)

test_df = test_df.copy()
test_df["prob"] = a2.flatten() #prob is probabilities of every game in the original dataset

correct = 0
total = 0

for yr in sorted(test_df["year"].unique()): #selects one data for example uhhh it will select 2001 for the first iteration (here it will select 2014 tho)
    yr_df = test_df[test_df["year"] == yr] # yr_df is the dataframe of only one year 
    if yr_df["Goty"].sum() == 0: #skip year if label=0, and total +1 hojayega
        continue
    total += 1
    best_idx = yr_df["prob"].idxmax() #picks up the index the game with highest probabilty in that year 
    is_correct = yr_df.loc[best_idx, "Goty"] == 1 #checks if true df.loc[row_index,column_index]
    if is_correct:
        correct += 1
    predicted = yr_df.loc[best_idx, "name"]
    actual = ", ".join(yr_df[yr_df["Goty"] == 1]["name"].values)
    print(f"{yr} {mark}  predicted: {predicted}  |  actual: {actual}")

print(f"\naccuracy: {correct}/{total} = {correct/total*100}%")




