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


#random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

features = [multifeat, vader,blob,tweetdir,tweet_vol,v_move,b_move]

feature = ['multifeat', 'vader','blob','tweetdir','tweet_vol','v_move','b_move']
for i in range(len(features)):
    scores = []
    mat = np.zeros([2,2])
    df_feature = pd.DataFrame(features[i])
    df_label = pd.DataFrame(diff)
    for j in range(1,10):
        print("Working on {} {} estimators".format(feature[i],j))
        #sbar = features[i].reshape(-1,1)

        if i == 0:
          #x_train, x_test, y_train, y_test = train_test_split(sbar, data["movement"], test_size=0.20, random_state=0)
          x_train, x_test, y_train, y_test = train_test_split(multifeat, df_label, test_size=0.20, random_state=0)
        else:
          x_train, x_test, y_train, y_test = train_test_split(df_feature, df_label, test_size=0.20, random_state=0)
        regressor = RandomForestClassifier(n_estimators=j*10, random_state=0,max_depth=i+2)
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        
        #predictions = knn.predict(x_test)
        score = regressor.score(x_test,y_test)
        #scores.append(score)
        print("Confusion Matrix is: \n{}".format(confusion_matrix(y_test,y_pred)))
        
        #print(classification_report(y_test,y_pred))
        print("Accuracy Score is: {}\n".format(round(accuracy_score(y_test, y_pred), 3)))
        f1 =  metrics.f1_score(y_test,y_pred)
        print("The F1 score is : {} \n".format(round(f1, 3)))
