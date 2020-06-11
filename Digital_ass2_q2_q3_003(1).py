# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:00:20 2019

@author: nicol
"""

import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import lda
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso





nat_geo=pd.read_csv("C:\\Users\\nicol\\Documents\\a_exams\\4_digital\\ass2\\natgeo.csv",encoding='utf8')


#------------------------ Task B --------------------------------
nat_geo.drop('Unnamed: 0', axis=1, inplace=True)
nat_geo.drop('tags', axis=1, inplace=True)

#normalize comments and likes
nat_geo['likes_count_norm']=(nat_geo['likes_count']-nat_geo['likes_count'].min())/(nat_geo['likes_count'].max()-nat_geo['likes_count'].min())
nat_geo['comments_count_norm']=(nat_geo['comments_count']-nat_geo['comments_count'].min())/(nat_geo['comments_count'].max()-nat_geo['comments_count'].min())

#calulate engagement rate
nat_geo['eng_rate'] = nat_geo['likes_count_norm']*.4 + nat_geo['comments_count_norm']*.6



#check blanks
test = nat_geo.isna().sum()
test2 = pd.DataFrame({'coulumn':test.index, 'blanks':test.values})


#manage the tags to create dummy columns
nat_geo['google_vision_tags'] = nat_geo['google_vision_tags'].map(lambda x: x.lstrip('[').rstrip(']'))
nat_geo['google_vision_tags'] = nat_geo['google_vision_tags'].map(lambda x: x.replace("'", ''))

testing = nat_geo['google_vision_tags'].str.split(',', expand=True)
tags = nat_geo['google_vision_tags']

tags_list = []
for i in range(0,len(tags)):
    for j in range(0,len(list(testing))):
        value = testing.iloc[i,j]
        if value not in tags_list:
            tags_list.append(value)
tags_list = list(filter(None, tags_list))
tags_list2 = [x.strip(' ') for x in tags_list]

#create dummy columns with tags - this operation takes about 40min
for elem in tags_list:
    nat_geo['tag_'+str(elem)] = ""

for i in tqdm(range(0, len(nat_geo))):
    value = nat_geo.iloc[i,4]
    for j in range(0, len(tags_list)):
        elem = tags_list[j]
        if elem in value:
            k = 8+j
            nat_geo.iloc[i,k] = 1
        else:
            k = 8+j
            nat_geo.iloc[i,k] = 0
#nat_geo.to_excel("C:\\Users\\nicol\\Documents\\a_exams\\4_digital\\ass2\\geo_with_tagDummies.xlsx",index=False)


#------- Modelling using TAGS ---------------------
#Start Modelling - Feature selection with LASSO and Random Forest
#Feature selection via LASSO
X_tags = nat_geo.iloc[:,8:]
Y_target = np.where(nat_geo['eng_rate']>nat_geo['eng_rate'].mean()+nat_geo['eng_rate'].std()/3,1,0)

model2 = Lasso(alpha=0.01, positive=True)
result2 = model2.fit(X_tags,Y_target)

result = pd.DataFrame(list(zip(X_tags.columns,model2.coef_)), columns = ['predictor','coef'])
result_sorted = result.sort_values(['coef'], ascending=[False])

#Feature selection via Random Forest
X_tags = nat_geo.iloc[:,8:]
Y_target = np.where(nat_geo['eng_rate']>nat_geo['eng_rate'].mean()+nat_geo['eng_rate'].std()/3,1,0)

randomforest = RandomForestClassifier(random_state=0)
model = randomforest.fit(X_tags,Y_target)

sfm = SelectFromModel(model, threshold=0.05)
sfm.fit(X_tags,Y_target)
    
result_rf = pd.DataFrame(list(zip(X_tags.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])
result_rf_sorted = result_rf.sort_values(['Gini coefficient'], ascending=[False])

#merge result and create final ranked list of predictors
list_rf = result_rf_sorted["predictor"].tolist()
list_lasso = result_sorted["predictor"].tolist()

predictor_list = []
rank_rf_list = []
rank_rf_lasso = []
for i in range(0,len(list_rf)-1):
    for j in range(0,len(list_lasso)-1):
       if list_rf[i] == list_lasso[j]:
           pred = list_rf[i]
           rank_rf = i
           rank_lasso = j
           predictor_list.append(pred)
           rank_rf_list.append(rank_rf)
           rank_rf_lasso.append(rank_lasso)
           break
       
dataframe = pd.DataFrame()
dataframe['predctor'] = predictor_list
dataframe['rank_rf'] = rank_rf_list
dataframe['rank_lasso'] = rank_rf_lasso


#calulate eng rate via logistic regression, iterating trough the list of best predictors to find the set 
#of predictors that maximizes the accuracy
number_predictors = []
accur_list = []
i_list = []
for i in range(4, 300):
    Extraction = dataframe[(dataframe.rank_rf <= i) | (dataframe.rank_lasso <= i)]
    list_pred = Extraction['predctor'].tolist()
    best_subset = nat_geo.loc[:, nat_geo.columns.str.contains('|'.join(list_pred))]
    column_names = list(best_subset)
    
    X_train_tag, X_test_tag, y_train_tag, y_test_tag = train_test_split(best_subset, Y_target, test_size = 0.33, random_state = 5)
    logistic_regression = LogisticRegression(random_state=0)
    reg_model = logistic_regression.fit(X_train_tag,y_train_tag)
    y_test_tag_pred = reg_model.predict(X_test_tag)
    acc_score = accuracy_score(y_test_tag, y_test_tag_pred)

    accur_list.append(acc_score)
    i_list.append(i)
    
res_reg = pd.DataFrame()
res_reg['predctor_number'] = i_list
res_reg['accuracy'] = accur_list
res_reg_sorted = res_reg.sort_values(['accuracy'], ascending=[False])   


#final list of predictors used
final_numb_pred = res_reg_sorted.iloc[0,0]
Extraction = dataframe[(dataframe.rank_rf <= final_numb_pred) | (dataframe.rank_lasso <= final_numb_pred)]
list_pred = Extraction['predctor'].tolist()
best_subset = nat_geo.loc[:, nat_geo.columns.str.contains('|'.join(list_pred))]

#run again the best model to get the y_test and y_test_pred
X_train_tag, X_test_tag, y_train_tag, y_test_tag = train_test_split(best_subset, Y_target, test_size = 0.33, random_state = 5)
logistic_regression = LogisticRegression(random_state=0)
reg_model = logistic_regression.fit(X_train_tag,y_train_tag)
y_test_tag_pred = reg_model.predict(X_test_tag)

#double check predicted number of true/false 
unique_elements, counts_elements = np.unique(y_test_tag_pred, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

#double check actual number of true/false 
unique_elements, counts_elements = np.unique(y_test_tag, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))


#final accuracy
final_accuracy = res_reg_sorted.iloc[0,1]
confusion_matrix(y_test_tag, y_test_tag_pred)




#--------- Model Using Caption ----------------------------
X_tags = nat_geo['caption']
Y_target = np.where(nat_geo['eng_rate']>nat_geo['eng_rate'].mean()+nat_geo['eng_rate'].std()/3,1,0)


X_train_caption, X_test_caption, y_train_caption, y_test_caption = train_test_split(X_tags, Y_target, test_size = 0.33, random_state = 5)
vectorizer = TfidfVectorizer(min_df=2, 
 ngram_range=(1,1), 
 stop_words='english', 
 strip_accents='unicode', 
 norm='l2')

X_train_caption = vectorizer.fit_transform(X_train_caption)
X_test_caption = vectorizer.transform(X_test_caption)

nb_classifier = MultinomialNB().fit(X_train_caption, y_train_caption)
y_nb_predicted = nb_classifier.predict(X_test_caption)


#double check predicted number of true/false 
unique_elements, counts_elements = np.unique(y_nb_predicted, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

#double check actual number of true/false 
unique_elements, counts_elements = np.unique(y_test_caption, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

# Accuracy of the regression model
accuracy_score(y_test_caption, y_nb_predicted)
confusion_matrix(y_test_caption, y_nb_predicted)




#--------- Model Using Both Tags and Caption ----------------------------
X_hybrid = pd.concat([nat_geo['caption'], best_subset], axis=1, ignore_index=False)
Y_target = np.where(nat_geo['eng_rate']>nat_geo['eng_rate'].mean()+nat_geo['eng_rate'].std()/3,1,0)

X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid = train_test_split(X_hybrid, Y_target, test_size = 0.33, random_state = 5)

X_train_desc = pd.DataFrame(vectorizer.fit_transform(X_train_hybrid["caption"]).toarray(), index = X_train_hybrid.index.values)
X_train_new = pd.concat([X_train_hybrid.iloc[:,1:], X_train_desc], axis=1, ignore_index=False)

X_test_desc = pd.DataFrame(vectorizer.transform(X_test_hybrid["caption"]).toarray(), index = X_test_hybrid.index.values)
X_test_new = pd.concat([X_test_hybrid.iloc[:,1:], X_test_desc], axis=1, ignore_index=True)

reg_model_hybrid = logistic_regression.fit(X_train_new,y_train_hybrid)
y_test_pred_hybrid = reg_model_hybrid.predict(X_test_new)

#double check predicted number of true/false 
unique_elements, counts_elements = np.unique(y_test_pred_hybrid, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

#double check actual number of true/false 
unique_elements, counts_elements = np.unique(y_test_hybrid, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

accuracy_score(y_test_hybrid, y_test_pred_hybrid)
confusion_matrix(y_test_hybrid, y_test_pred_hybrid)



# ------------------- Task C ---------------------------------------------
list(nat_geo)

image_id = 'image_id'
image_review = 'caption'
ntopics= 5;


word_tokenizer=RegexpTokenizer(r'\w+')
wordnet_lemmatizer = WordNetLemmatizer()
stopwords_nltk=set(stopwords.words('english'))

def tokenize_text(version_desc):
    lowercase=version_desc.lower()
    text = wordnet_lemmatizer.lemmatize(lowercase)
    tokens = word_tokenizer.tokenize(text)
    return tokens

vec_words = CountVectorizer(tokenizer=tokenize_text,stop_words=stopwords_nltk,decode_error='ignore')
total_features_words = vec_words.fit_transform(nat_geo[image_review])

print(total_features_words.shape)

model = lda.LDA(n_topics=int(ntopics), n_iter=500, random_state=1)
model.fit(total_features_words)

topic_word = model.topic_word_ 
doc_topic=model.doc_topic_
doc_topic=pd.DataFrame(doc_topic)
nat_geo=nat_geo.join(doc_topic)
geo_pictures=pd.DataFrame()

for i in range(int(ntopics)):
    topic="topic_"+str(i)
    geo_pictures[topic]=nat_geo.groupby([image_id])[i].mean()

geo_pictures=geo_pictures.reset_index()
topics=pd.DataFrame(topic_word)
topics.columns=vec_words.get_feature_names()
topics1=topics.transpose()
#topics1.to_excel("C:\\Users\\nicol\\Documents\\a_exams\\4_digital\\ass2\\topic_word_dist_geo.xlsx")
#geo_pictures.to_excel("C:\\Users\\nicol\\Documents\\a_exams\\4_digital\\ass2\\geo_pict_topic_dist.xlsx",index=False)


topics1.columns = ['Topic_0', 'Topic_1', 'Topic_2', 'Topic_3', 'Topic_4']
                   

#find the best 15 words per each topic
sort_topic0 = topics1.sort_values(['Topic_0'], ascending=[False])
list_words_t0 = list(sort_topic0.index.values) 
list_words_t0 = list_words_t0[0:15]

sort_topic1 = topics1.sort_values(['Topic_1'], ascending=[False])
list_words_t1 = list(sort_topic1.index.values) 
list_words_t1 = list_words_t1[0:15]

sort_topic2 = topics1.sort_values(['Topic_2'], ascending=[False])
list_words_t2 = list(sort_topic2.index.values) 
list_words_t2 = list_words_t2[0:15]

sort_topic3 = topics1.sort_values(['Topic_3'], ascending=[False])
list_words_t3 = list(sort_topic3.index.values) 
list_words_t3 = list_words_t3[0:15]

sort_topic4 = topics1.sort_values(['Topic_4'], ascending=[False])
list_words_t4 = list(sort_topic4.index.values) 
list_words_t4 = list_words_t4[0:15]

#create a final list of best words
final_list = list_words_t0 + list_words_t1 + list_words_t2 + list_words_t3 + list_words_t4
final_list_unique = list(set(final_list))
len(final_list_unique)

#for each best word, calulate the rabnking within each topic
list_main = []
list_t0 = []
list_t1 = []
list_t2 = []
list_t3 = []
list_t4 = []
for i in range(0,len(final_list_unique)):
    count = 0
    for j in range(0,len(list_words_t0)):
        if final_list_unique[i] == list_words_t0[j]:
            index_t0 = j+1
            count = 1
    if count == 0:
        index_t0 = 0 
    count = 0
    for j in range(0,len(list_words_t1)):
        if final_list_unique[i] == list_words_t1[j]:
            index_t1 = j+1
            count = 1
    if count == 0:
        index_t1 = 0 
    count = 0    
    for j in range(0,len(list_words_t2)):
        if final_list_unique[i] == list_words_t2[j]:
            index_t2 = j+1
            count = 1
    if count == 0:
        index_t2 = 0 
    count = 0        
    for j in range(0,len(list_words_t3)):
        if final_list_unique[i] == list_words_t3[j]:
            index_t3 = j+1
            count = 1
    if count == 0:
        index_t3 = 0 
    count = 0       
    for j in range(0,len(list_words_t4)):
        if final_list_unique[i] == list_words_t4[j]:
            index_t4 = j+1
            count = 1
    if count == 0:
        index_t4 = 0 
    
    list_t0.append(index_t0)
    list_t1.append(index_t1)
    list_t2.append(index_t2)
    list_t3.append(index_t3)
    list_t4.append(index_t4)    
    list_main.append(final_list_unique[i])
 
len(list_main)    

result = pd.DataFrame()
result['word'] = list_main
result['ranking_t0'] = list_t0
result['ranking_t1'] = list_t1
result['ranking_t2'] = list_t2
result['ranking_t3'] = list_t3
result['ranking_t4'] = list_t4
result.to_excel("C:\\Users\\nicol\\Documents\\a_exams\\4_digital\\ass2\\ranking_cross_topics_01.xlsx")


col = list(nat_geo)
topic_df = nat_geo[nat_geo.columns[-5:]]
topic_df.columns = ['Topic_0', 'Topic_1', 'Topic_2', 'Topic_3', 'Topic_4']
picture_id_eng_rate = nat_geo[['image_id', 'eng_rate']]

topic_analysis = pd.concat([picture_id_eng_rate, topic_df], axis=1, ignore_index=False)
topic_analysis_sorted = topic_analysis.sort_values(['eng_rate'], ascending=[False])

list(best_subset)
rating = topic_analysis_sorted['image_id'].tolist()
quart = int(len(rating)/4)
top_quart = rating[:quart]
best_subset = topic_analysis_sorted[topic_analysis_sorted['image_id'].isin(top_quart)]
avg_topic0_best = best_subset['Topic_0'].mean()
avg_topic1_best = best_subset['Topic_1'].mean()
avg_topic2_best = best_subset['Topic_2'].mean()
avg_topic3_best = best_subset['Topic_3'].mean()
avg_topic4_best = best_subset['Topic_4'].mean()
avg_best = [avg_topic0_best, avg_topic1_best, avg_topic2_best, avg_topic3_best, avg_topic4_best]

low_quart_ix = len(rating)-quart
ws_quart = rating[low_quart_ix:]
worst_subset = topic_analysis_sorted[topic_analysis_sorted['image_id'].isin(ws_quart)]
avg_topic0_worst = worst_subset['Topic_0'].mean()
avg_topic1_worst = worst_subset['Topic_1'].mean()
avg_topic2_worst = worst_subset['Topic_2'].mean()
avg_topic3_worst = worst_subset['Topic_3'].mean()
avg_topic4_worst = worst_subset['Topic_4'].mean()
avg_worst = [avg_topic0_worst, avg_topic1_worst, avg_topic2_worst, avg_topic3_worst, avg_topic4_worst]


coulmns_fdf = ['Topic_0', 'Topic_1', 'Topic_2', 'Topic_3', 'Topic_4']
index_fdf = ['best_quartile', 'worst_quartile']
final_result = pd.DataFrame(columns=coulmns_fdf, index=index_fdf)

for i in range(0,len(list(final_result))):
    final_result.iloc[0,i] = avg_best[i]
for i in range(0,len(list(final_result))):
    final_result.iloc[1,i] = avg_worst[i]






