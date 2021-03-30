import os
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, ClassifierMixin, MultiOutputClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from utils import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#加载数据
label= np.array(pd.read_csv('./data/label_dropout.csv'))
#train_sampel = pd.read_csv('./data/train_sample_500.csv')
train_sampel = pd.read_csv('./data/train_sample_shuffle_dropout.csv')
test_sampel = pd.read_csv('./data/test_sample_shuffle_dropout.csv')
#test_sampel = pd.read_csv('./data/test_sample_500.csv')
#数据归一化
stdScalar = StandardScaler()
train_df = stdScalar.fit_transform(np.float_(train_sampel))
test_df = stdScalar.fit_transform(np.float_(test_sampel))


losslist = []
nfold = 5
kf = MultilabelStratifiedKFold(n_splits=nfold, shuffle=True, random_state=2020)
lr_oof = np.zeros(label.shape)
# 存储测试集的概率
probility = np.zeros((len(test_df), label.shape[1]))
i = 0
model_type = 'ensemble'
# model_type ='single'
# K折交叉划分训练
for train_index, valid_index in kf.split(train_df, label):
    print("\nFold {}".format(i + 1))
    i += 1
    X_train, label_train = train_df[train_index], label[train_index]
    X_valid, label_valid = train_df[valid_index], label[valid_index]
    
    clf1 = OneVsRestClassifier(XGBClassifier(eval_metric= 'mlogloss',use_label_encoder=False,n_estimators=150))
    clf2 = LGBMClassifier()
    clf3 = LogisticRegression(max_iter =500, n_jobs=20)
    # 模型集成方法1
    if model_type == 'ensemble':
        # 因为XGB的单模型效果比其他两个好，所以权重是2:1:1
        model = OneVsRestClassifier(EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],weights=[2, 1, 1], voting='soft', verbose=2))
    # 模型集成方法2
    elif model_type == 'stacking':
        lr = LogisticRegression()
        base = StackingClassifier(classifiers=[clf1, clf2, clf3],use_probas=True,average_probas=False, meta_classifier=lr,verbose=2)
        model = OneVsRestClassifier(base)
    else:
        # 单模型训练
        model = OneVsRestClassifier(XGBClassifier(eval_metric= 'mlogloss',use_label_encoder=False,n_estimators=150))
    model.fit(X_train, label_train)
    # 计算损失
    lr_oof[valid_index] = model.predict_proba(X_valid,)
    loss = Mutilogloss(label_valid[:,:-1,], lr_oof[valid_index][:,:-1,])
    losslist.append(loss)
    # 多个fold的预测结果叠加
    probility += model.predict_proba(test_df) / nfold
    print(losslist)
    
print(np.mean(losslist))

# 保存存提交数据
submit_dir='submits/'
if not os.path.exists(submit_dir): os.makedirs(submit_dir)
str_w=''
with open(submit_dir+'machine_model_submit.csv','w') as f:
    for i in range(len(probility)):
        list_to_str = [str(x) for x in list(probility[i])][0:-1]
        liststr = " ".join(list_to_str)
        str_w+=str(i)+'|'+','+'|'+liststr+'\n'
    str_w=str_w.strip('\n')
    f.write(str_w)
print()
