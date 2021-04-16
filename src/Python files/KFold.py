from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold # import KFold
from sklearn import metrics
import matplotlib.pyplot as plt

data =pd.read_csv("bitcoin_and_sentiments_data.csv")
data.columns
data.fillna(inplace=True, value=0)

# Vader Sentiment List data
vader = data["vader_sent_mean_by_hour"].values

# Sentiment Data through TextBlob
blob = data["blob_sent_mean_by_hour"].values

# Change in twitter direction if less or more tweets are happening
tweetdir = data["tweet_diff"].values

# Volume of tweets in an hour
tweet_vol = data["Daily_Weight_count_by_hour"].values

# Price movement direction in 0 for price going down and 1 for price moving up
diff = data['movement'].values

# Multi feature dataset for training the model
multi_feature = data.filter(["vader_sent_mean_by_hour","blob_sent_mean_by_hour","tweet dir","Daily_Weight_count_by_hour"])

# Bitcoin price every hour
close = data['closed_price_by_hour'].values

def f1(cm):
    aNo = cm[0][0]
    ayes = cm[1][0]
    pno = cm[0][1]
    pyes = cm[1][1]


#Kfold Cross Validation with Logistic Regression

features = [multi_feature, vader,blob,tweetdir,tweet_vol] #List of features to be tried in KFold Logistic regression

feature = ['multi_feature', 'vader','blob','tweet_dir','tweet_volume']

for i in range(len(features)):
    if i >0:
        xbar = features[i].reshape(-1,1)
    else:
        xbar = features[i]
    kf = KFold(n_splits=10) # Define the split - into 2 folds 
    
    kf.get_n_splits(xbar) # returns the number of splitting iterations in the cross-validator
    #KFold(n_splits=10, random_state=None, shuffle=False)
    scores = []
    f1=[]
    mat = np.zeros([2,2])
    for train_index, test_index in kf.split(xbar):
        X_train, X_test = xbar[train_index[0]:train_index[1189]], xbar[test_index[0]:test_index[132]]
        y_train, y_test = diff[train_index[0]:train_index[1189]], diff[test_index[0]:test_index[132]]
        logisticRegr = LogisticRegression()
        logisticRegr.fit(X_train, y_train)

        predictions = logisticRegr.predict(X_test)
        score = logisticRegr.score(X_test, y_test)
        scores.append(score)
        cm = metrics.confusion_matrix(y_test, predictions)
        f1_score = metrics.f1_score(y_test,predictions)
        f1.append(f1_score)
        mat += cm
        
    idx = np.mean(scores)
    print("{} Score : {}".format(feature[i],round(idx, 3)))
    print("{} F1 Score : {}".format(feature[i],round(np.mean(f1), 3)))
    print(mat)
