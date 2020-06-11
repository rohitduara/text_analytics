import pandas as pd

filepath = "/Users/akkuanu/Desktop/Social Media Analysis/Assignment 1/train.csv"
train_df = pd.read_csv(filepath)


def dataTransformation(df, cols_to_subtract, cols_to_divide):
    mod_train_df = pd.DataFrame()
    ### Data trasformation. Finding difference for each attribute type
    mod_train_df['Choice'] = df['Choice']
    for item in cols_to_subtract:
        mod_train_df["diff_"+item] = df['A_'+item] - df['B_'+item]
    for item in cols_to_divide:
        mod_train_df["ratio_"+item] = df['A_'+item] / df['B_'+item]
    return(mod_train_df)

cols_to_subtract = ["follower_count", "following_count", "listed_count", "network_feature_1", "network_feature_2", "network_feature_3"]
cols_to_divide = ["mentions_received","retweets_received","mentions_sent", "retweets_sent", "posts"]

mod_train_df = dataTransformation(train_df, cols_to_subtract, cols_to_divide)

#from sklearn.linear_model import LogisticRegression
y = mod_train_df['Choice']
X = mod_train_df.iloc[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

import pandas as pd
feature_imp = pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
feature_imp

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

#### Remove insignificant predictors (<.08 in feature importance): "Best" model 
insignificant_features = feature_imp[feature_imp<.09].keys()
X_train.drop(insignificant_features, axis=1, inplace=True)
X_test.drop(insignificant_features, axis=1, inplace=True)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
model_accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:",model_accuracy)

feature_imp = pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
feature_imp

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


test_data = train_df.ix[X_test.index]
test_data["Predicted_Choice"] = y_pred
#### Financial Calculations:

Profit_per_purchase = 10
Actual_followers_of_influencers = test_data[test_data.Choice == 1].B_follower_count.sum() + test_data[test_data.Choice == 0].A_follower_count.sum()

## Without analytics
perc_of_purchase = 0.0001 #0.01%

## Without analytics
Profit_without_analytics = (perc_of_purchase * Profit_per_purchase * Actual_followers_of_influencers)
## With perfect model
perc_of_purchase_with_two_tweets = 0.00015
Profit_with_perfect_analytics = (perc_of_purchase_with_two_tweets * Profit_per_purchase * Actual_followers_of_influencers)
#Profit_with_perfect_analytics = (perc_of_purchase_with_two_tweets * Profit_per_purchase * Actual_followers_of_influencers) -  Total_Cost

# With our model
Profit_with_our_analytics = (model_accuracy * perc_of_purchase_with_two_tweets * Profit_per_purchase * Actual_followers_of_influencers)


## % increase with perfect model
print("Percentage increase with perfect analytics as compared to no analytics:",(Profit_with_perfect_analytics/Profit_without_analytics - 1)*100, "%")
## % increase with our model
print("Percentage increase with our analytics model as compared to no analytics:",(Profit_with_our_analytics/Profit_without_analytics - 1)*100, "%")