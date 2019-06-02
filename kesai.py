# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:17:10 2019

@author: javaprison
"""

import pandas as pd
import gensim
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing, decomposition
from sklearn.svm import SVC
import xgboost as xgb
data_train=pd.read_csv(r'train.csv',lineterminator='\n')
data_test=pd.read_csv(r'test.csv',lineterminator='\n')
x=data_train.review
x1=data_test.review
y=data_train.label
class_mapping={'Positive':1,'Negative':0}
x_train=[]
x_test=[]
def review2wordlist(review):
    review = re.sub(r"[^a-zA-Z]", " ", review)
    wordlist = review.lower().split()
    return wordlist
for i in range(len(x)):
    words = review2wordlist(x[i])
    x_train.append(" ".join(words))
for i in range(len(x1)):
    words = review2wordlist(x1[i])
    x_test.append(" ".join(words))
none_size=0
for i in range(len(x_train)):
    if(x_train[i]==""):
        print(i)
        none_size=none_size+1
print("要删除的评论是")
print(x[5256])
print(x_train[5256])
print("它的标签是")
print(y[5256])
del x_train[5256]
del y[5256]
print("______________________")
print("要删除的评论是")
print(x[5999])
print(x_train[5998])
print("它的标签是")
print(y[5999])
del x_train[5998]
del y[5999]
print("______________________")
print("______________________")
print("______________________")
y_train=y.map(class_mapping).values
#定义一个评价标准
def multiclass_logloss(actual, predicted, eps=1e-15):
    """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
    :param actual: 包含actual target classes的数组
    :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
#普通tfidf
def data_tfidf(x):
    vectorizer=CountVectorizer()
    transformer = TfidfTransformer()
    x = transformer.fit_transform(vectorizer.fit_transform(x))
    return x
#加一些处理的tfidf
def data_tfidf2(x):
    tfv = TFIV(min_df=3,max_df=0.5, max_features=None,
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        )
    tfv.fit(x)
    x=tfv.transform(x)
    return x
#word2vec处理
model = gensim.models.Word2Vec(x, min_count =1, window =8, size=50)
def sent2vec(x):
    M=[]
    #embeddings_index = dict(zip(model.wv.index2word, model.wv.vectors))
    for w in x:
        try:
            M.append(model[w])            # 将词向量矩阵添加到列表 M 当中
        except:
            continue
        M = np.array(M)                       # 转换为数组
        v = M.sum(axis=0)
        sqrt_value = (v ** 2).sum()
        if type(v) != np.ndarray:
            return np.zeros(300).toarray()
        return v / np.sqrt(sqrt_value)
x_train1=[]
for x in range(len(x_train)):
    x_train1.append((sent2vec((x_train[x]))).tolist())
#svm数据处理
x_all_tfidf=x_train+x_test
x_all_tfidf=data_tfidf(x_all_tfidf)
x_train_tfidf=x_all_tfidf[:(len(x_train))]
x_test_tfidf=x_all_tfidf[(len(x_train)):]
x_train_tfidf,x_valid_tfidf,y_train_tfidf,y_valid_tfidf=train_test_split(x_train_tfidf,y_train,test_size=0.2,random_state=42)
svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(x_train_tfidf)
x_train_svd = svd.transform(x_train_tfidf)
x_valid_svd = svd.transform(x_valid_tfidf)
x_test_svd=svd.transform(x_test_tfidf)
scl = preprocessing.StandardScaler()
scl.fit(x_train_svd)
x_train_svd_scl = scl.transform(x_train_svd)
x_valid_svd_scl = scl.transform(x_valid_svd)
x_test_svd_scl= scl.transform(x_test_svd)
#普通tfidf datas
x_all_tfidf=x_train+x_test
x_all_tfidf=data_tfidf(x_all_tfidf)
x_train_tfidf=x_all_tfidf[:(len(x_train))]
x_test_tfidf=x_all_tfidf[(len(x_train)):]
x_train_tfidf,x_valid_tfidf,y_train_tfidf,y_valid_tfidf=train_test_split(x_train_tfidf,y_train,test_size=0.2,random_state=42)
#特殊处理tfidf datas
x_all_tfidf2=x_train+x_test
x_all_tfidf2=data_tfidf2(x_all_tfidf2)
x_train_tfidf2=x_all_tfidf2[:(len(x_train))]
x_test_tfidf2=x_all_tfidf2[(len(x_train)):]
x_train_tfidf2,x_valid_tfidf2,y_train_tfidf2,y_valid_tfidf2=train_test_split(x_train_tfidf2,y_train,test_size=0.2,random_state=42)
#word2vec datas
x_test_w2v=[]
for x in range(len(x_test)):
    x_test_w2v.append((sent2vec((x_test[x]))).tolist())
x_train_w2v, x_valid_w2v, y_train_w2v, y_valid_w2v = train_test_split(x_train1,y_train,test_size=0.2,random_state=42)
#lr in tfidf
lr = LogisticRegression(C=30)
grid_value = {'solver':['sag','liblinear','lbfgs']}
model_lr = GridSearchCV(lr, cv=20, scoring='roc_auc', param_grid=grid_value)
model_lr.fit(x_train_tfidf, y_train_tfidf)
lr_valid_result = model_lr.predict_proba(x_valid_tfidf)
lr_validresult=[]
for i in range(len(lr_valid_result)):
    po=lr_valid_result[i][1]
    lr_validresult.append(po)
print("lr in tfidf")
scores=metrics.roc_auc_score(y_valid_tfidf,lr_validresult)
print("auc score")
print(scores)
print("accuracy :")
print(model_lr.score(x_train_tfidf,y_train_tfidf))
if scores>0.85:
    lr_test_result=model_lr.predict_proba(x_test_tfidf)
    lr_testresult=[]
    for i in range(len(lr_test_result)):
        po=lr_test_result[i][1]
        lr_testresult.append(po)
    lr_df = pd.DataFrame({'ID':data_test.ID, 'Pred':lr_testresult})
    lr_df.to_csv("LR_result_tfidf.csv", index=False)
print("______________________")
print("______________________")
#sgd in word2vec
model_sgd = SGDClassifier(loss='modified_huber')
model_sgd.fit(x_train_w2v, y_train_w2v)
sgd_result_w2v = model_sgd.predict_proba(x_valid_w2v)
sgdresult=[]
for i in range(len(sgd_result_w2v)):
    po=sgd_result_w2v[i][1]
    sgdresult.append(po)
print("sgd in word2vec")
scores=metrics.roc_auc_score(y_valid_w2v,sgdresult)
print("auc score")
print(scores)
print("accuracy :")
print(model_sgd.score(x_train_w2v,y_train_w2v))
if scores>0.85:
    sgd_test_result=model_sgd.predict_proba(x_test_w2v)
    sgd_testresult=[]
    for i in range(len(sgd_test_result)):
        po=sgd_test_result[i][1]
        sgd_testresult.append(po)
    lr_df = pd.DataFrame({'ID':data_test.ID, 'Pred':sgd_testresult})
    lr_df.to_csv("sgd_result_w2v.csv", index=False)
print("______________________")
print("______________________")
#lr in tfidf2
lr = LogisticRegression(C=30)
grid_value = {'solver':['sag','liblinear','lbfgs']}
model_lr = GridSearchCV(lr, cv=20, scoring='roc_auc', param_grid=grid_value)
model_lr.fit(x_train_tfidf2, y_train_tfidf2)
lr_valid_result = model_lr.predict_proba(x_valid_tfidf2)
lr_validresult=[]
for i in range(len(lr_valid_result)):
    po=lr_valid_result[i][1]
    lr_validresult.append(po)
print("lr in tfidf2")
scores=metrics.roc_auc_score(y_valid_tfidf2,lr_validresult)
print("auc score")
print(scores)
print("accuracy :")
print(model_lr.score(x_train_tfidf2,y_train_tfidf2))
if scores>0.85:
    lr_test_result=model_lr.predict_proba(x_test_tfidf2)
    lr_testresult=[]
    for i in range(len(lr_test_result)):
        po=lr_test_result[i][1]
        lr_testresult.append(po)
    lr_df = pd.DataFrame({'ID':data_test.ID, 'Pred':lr_testresult})
    lr_df.to_csv("LR_result_tfidf2.csv", index=False)
print("______________________")
print("______________________")
#lr in word2vec
lr = LogisticRegression(C=30)
grid_value = {'solver':['sag','liblinear','lbfgs']}
model_lr = GridSearchCV(lr, cv=20, scoring='roc_auc', param_grid=grid_value)
model_lr.fit(x_train_w2v, y_train_w2v)
lr_valid_result = model_lr.predict_proba(x_valid_w2v)
lr_validresult=[]
for i in range(len(lr_valid_result)):
    po=lr_valid_result[i][1]
    lr_validresult.append(po)
print("lr in w2v")
scores=metrics.roc_auc_score(y_valid_w2v,lr_validresult)
print("auc score")
print(scores)
print("accuracy :")
print(model_lr.score(x_train_w2v,y_train_w2v))
if scores>0.85:
    lr_test_result=model_lr.predict_proba(x_test_w2v)
    lr_testresult=[]
    for i in range(len(lr_test_result)):
        po=lr_test_result[i][1]
        lr_testresult.append(po)
    lr_df = pd.DataFrame({'ID':data_test.ID, 'Pred':lr_testresult})
    lr_df.to_csv("LR_result_w2v.csv", index=False)
print("______________________")
print("______________________")
#sgd in tfidf
model_sgd = SGDClassifier(loss='modified_huber')
model_sgd.fit(x_train_tfidf, y_train_tfidf)
sgd_result_tfidf = model_sgd.predict_proba(x_valid_tfidf)
sgdresult=[]
for i in range(len(sgd_result_tfidf)):
    po=sgd_result_tfidf[i][1]
    sgdresult.append(po)
print("sgd in tfidf")
scores=metrics.roc_auc_score(y_valid_tfidf,sgdresult)
print("auc score")
print(scores)
print("accuracy :")
print(model_sgd.score(x_train_tfidf,y_train_tfidf))
if scores>0.85:
    sgd_test_result=model_sgd.predict_proba(x_test_tfidf)
    sgd_testresult=[]
    for i in range(len(sgd_test_result)):
        po=sgd_test_result[i][1]
        sgd_testresult.append(po)
    lr_df = pd.DataFrame({'ID':data_test.ID, 'Pred':sgd_testresult})
    lr_df.to_csv("sgd_result_tfidf.csv", index=False)
print("______________________")
print("______________________")
#sgd in tfidf2
model_sgd = SGDClassifier(loss='modified_huber')
model_sgd.fit(x_train_tfidf2, y_train_tfidf2)
sgd_result_tfidf2 = model_sgd.predict_proba(x_valid_tfidf2)
sgdresult=[]
for i in range(len(sgd_result_tfidf2)):
    po=sgd_result_tfidf2[i][1]
    sgdresult.append(po)
print("sgd in tfidf2")
scores=metrics.roc_auc_score(y_valid_tfidf2,sgdresult)
print("auc score")
print(scores)
print("accuracy :")
print(model_sgd.score(x_train_tfidf2,y_train_tfidf2))
if scores>0.85:
    sgd_test_result=model_sgd.predict_proba(x_test_tfidf2)
    sgd_testresult=[]
    for i in range(len(sgd_test_result)):
        po=sgd_test_result[i][1]
        sgd_testresult.append(po)
    lr_df = pd.DataFrame({'ID':data_test.ID, 'Pred':sgd_testresult})
    lr_df.to_csv("sgd_result_tfidf2.csv", index=False)
print("______________________")
print("______________________")
#mnb in tfidf
model_nb = MultinomialNB()
model_nb.fit(x_train_tfidf, y_train_tfidf)
nb_result_tfidf = model_nb.predict_proba(x_valid_tfidf)
nbresult=[]
for i in range(len(nb_result_tfidf)):
    po=nb_result_tfidf[i][1]
    nbresult.append(po)
print("nb in tfidf")
scores=metrics.roc_auc_score(y_valid_tfidf,nbresult)
print("auc score")
print(scores)
print("accuracy :")
print(model_nb.score(x_train_tfidf,y_train_tfidf))
if scores>0.85:
    nb_test_result=model_nb.predict_proba(x_test_tfidf)
    nb_testresult=[]
    for i in range(len(nb_test_result)):
        po=nb_test_result[i][1]
        nb_testresult.append(po)
    lr_df = pd.DataFrame({'ID':data_test.ID, 'Pred':nb_testresult})
    lr_df.to_csv("nb_result_tfidf.csv", index=False)
print("______________________")
print("______________________")
#mnb in tfidf2
model_nb = MultinomialNB()
model_nb.fit(x_train_tfidf2, y_train_tfidf2)
nb_result_tfidf2 = model_nb.predict_proba(x_valid_tfidf2)
nbresult=[]
for i in range(len(nb_result_tfidf2)):
    po=nb_result_tfidf2[i][1]
    nbresult.append(po)
print("nb in tfidf2")
scores=metrics.roc_auc_score(y_valid_tfidf2,nbresult)
print("auc score")
print(scores)
print("accuracy :")
print(model_nb.score(x_train_tfidf2,y_train_tfidf2))
if scores>0.85:
    nb_test_result=model_nb.predict_proba(x_test_tfidf2)
    nb_testresult=[]
    for i in range(len(nb_test_result)):
        po=nb_test_result[i][1]
        nb_testresult.append(po)
    lr_df = pd.DataFrame({'ID':data_test.ID, 'Pred':nb_testresult})
    lr_df.to_csv("nb_result_tfidf2.csv", index=False)
print("______________________")
print("______________________")
#nb in word2vec
model_nb = GaussianNB()
model_nb.fit(x_train_w2v, y_train_w2v)
nb_result_w2v = model_nb.predict_proba(x_valid_w2v)
nbresult=[]
for i in range(len(nb_result_w2v)):
    po=nb_result_w2v[i][1]
    nbresult.append(po)
print("nb in w2v")
scores=metrics.roc_auc_score(y_valid_w2v,nbresult)
print("auc score")
print(scores)
print("accuracy :")
print(model_nb.score(x_train_w2v,y_train_w2v))
if scores>0.85:
    nb_test_result=model_nb.predict_proba(x_test_w2v)
    nb_testresult=[]
    for i in range(len(nb_test_result)):
        po=nb_test_result[i][1]
        nb_testresult.append(po)
    lr_df = pd.DataFrame({'ID':data_test.ID, 'Pred':nb_testresult})
    lr_df.to_csv("nb_result_w2v.csv", index=False)
print("______________________")
print("______________________")
#svm
model_svm = SVC(C=1.0, probability=True) # since we need probabilities
model_svm.fit(x_train_svd_scl, y_train_tfidf)
svc_result = model_svm.predict_proba(x_valid_svd_scl)
svcresult=[]
for i in range(len(svc_result)):
    po=svc_result[i][1]
    svcresult.append(po)
print("svm")
scores=metrics.roc_auc_score(y_valid_tfidf,svcresult)
print("auc score")
print(scores)
print("accuracy :")
print(model_svm.score(x_train_svd_scl, y_train_tfidf))
if scores>0.85:
    svc_test_result=model_svm.predict_proba(x_test_svd_scl)
    svc_testresult=[]
    for i in range(len(svc_test_result)):
        po=svc_test_result[i][1]
        svc_testresult.append(po)
    lr_df = pd.DataFrame({'ID':data_test.ID, 'Pred':svc_testresult})
    lr_df.to_csv("svm.csv", index=False)
print("______________________")
print("______________________")
#xgboost
model_xg=xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                        subsample=0.8, nthread=10, learning_rate=0.1)
model_xg.fit(x_train_tfidf.tocsc(),y_train_tfidf)
xg_result=model_xg.predict_proba(x_valid_tfidf.tocsc())
xgresult=[]
for i in range(len(xg_result)):
    po=xg_result[i][1]
    xgresult.append(po)
print("xgboot")
scores=metrics.roc_auc_score(y_valid_tfidf,xgresult)
print("auc score")
print(scores)
print("accuracy :")
print(model_xg.score(x_train_tfidf.tocsc(),y_train_tfidf))
if scores>0.85:
    xg_test_result=model_xg.predict_proba(x_test_tfidf.tocsc())
    xg_testresult=[]
    for i in range(len(xg_test_result)):
        po=xg_test_result[i][1]
        xg_testresult.append(po)
    lr_df = pd.DataFrame({'ID':data_test.ID, 'Pred':xg_testresult})
    lr_df.to_csv("xgboost.csv", index=False)
print("______________________")
print("______________________")