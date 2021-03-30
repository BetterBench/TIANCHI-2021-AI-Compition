import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, ClassifierMixin, MultiOutputClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.metrics import roc_auc_score
#加载数据
label= np.array(pd.read_csv('./data/label.csv'))
train = pd.read_csv('./temp/train.csv',header = None,names=['id','text','label'])

def adjust_model():
    Tdf = TfidfVectorizer(ngram_range=(1,2),max_features=500)
    tdf_data = Tdf.fit_transform(train['text'])
    X_train,X_test,y_train,y_test = train_test_split(tdf_data,label,test_size=0.3)
    paralist = []
    score_dict = {"list_n":[],"list_f":[],"loss":[]}
    # for n in paralist
    param_test1 = {'estimator__max_depth':range(2,8,2)}
    model = OneVsRestClassifier(XGBClassifier(eval_metric= 'mlogloss',use_label_encoder=False,n_estimators=150))
    model.fit(X_train, y_train)

    predict = model.predict_proba(X_test)
    score = roc_auc_score(y_test,predict)
    print(score)
    print()
    # gsearch1 = GridSearchCV(model,param_grid = param_test1,scoring='roc_auc',n_jobs=100, cv=5,verbose=2)
    # gsearch1.fit(X_train, y_train)
    # print("参数\n",gsearch1.best_params_)
    # print("最佳得分",gsearch1.best_score_)
    '''
    XGB 
    {'estimator__max_depth': 9, 'estimator__min_child_weight': 1}
     {'estimator__max_depth': 11}0.9812110365828264
      {'estimator__n_estimators': 150} 0.9834881407453535
    调参后：0.9726861215062805

    '''
    '''
    LGB
     {'estimator__max_depth': 6}最佳得分 0.9811430144134826
    '''
    print()
adjust_model()

# 调参tfidf的ngram大小和featue的维度
# 得出最佳是ngram=(1,2),feature = 500
def adjust_idtdf():
    
    list_ngram = [1,2,3,4,5]
    list_feature = [100,200,300,400,500]
    #分数记录字典
    score_dict = {"list_n":[],"list_f":[],"loss":[]}
    #创建方法进行验证
    def para_Tdf(data_x):
        for n in list_ngram:
            for fea in list_feature:
                Tdf = TfidfVectorizer(ngram_range=(1,n),max_features=fea)
                tdf_data = Tdf.fit_transform(data_x)
                # tdf_data = tdf_data.toarray()
                X_train,X_test,y_train,y_test = train_test_split(tdf_data,label,test_size=0.3)
                model = OneVsRestClassifier(XGBClassifier(eval_metric= 'mlogloss',use_label_encoder=False,n_estimators=50))
                model.fit(X_train, y_train)
                predict = model.predict_proba(X_test)
                loss = Mutilogloss(y_test,predict)
                score_dict["list_n"].append(n)
                score_dict['list_f'].append(fea)
                score_dict['loss'].append(loss)
                print("n={0},feature={1},loss={2}".format(n,fea,loss))
    #方法调用
    para_Tdf(train['text'])
    #以DataFrame形式显示分数
    print(score_dict)


