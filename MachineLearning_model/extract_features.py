import pandas as pd
import numpy as np
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ =="__main__":
    test_data = pd.read_csv('./data/track1_round1_testA_20210222.csv',header=None, names=['id', 'text'])
    train_data = pd.read_csv('./data/track1_round1_train_20210222.csv',header=None, names=['id', 'text','label'])
    # 删除高频字符，去除首尾的”|“
    print("正在去除高频字符...")
    train_aug_data = pd.DataFrame()
    train_aug_data['label'] = train_data['label']
    # 删除高频字符：693 328 可能是标点符号
    train_aug_data['text']= delete_highfrequency_word(train_data)
    test_data['text'] = delete_highfrequency_word(test_data)

    # 合并，准备提取tfidf特征
    print(train_aug_data.shape, test_data.shape)
    df = pd.concat([train_aug_data, test_data], axis=0, ignore_index=True)
    print("正在提取Text特征...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2),max_features=500)
    tfidf_feature = tfidf.fit_transform(df['text'])
    svd_feature = tfidf_feature.toarray()
    # 拆分出训练集和测试集
    train_data_sample = svd_feature[:-3000]
    test_data_sample = svd_feature[-3000:]
    # 统计句子长度，作为一个特征
    print("训练集-正在计算句子长度...")
    train_sample = pd.DataFrame(train_data_sample)
    train_sample.columns = ['tfidf_'+str(i) for i in train_sample.columns]
    train_sample['textlen'] = count_text_len(train_aug_data)
    print("测试集-正在统计句子长度...")
    test_sample = pd.DataFrame(test_data_sample)
    test_sample.columns = ['tfidf_'+str(i) for i in test_sample.columns]
    test_sample['textlen'] = count_text_len(test_data)
    #label编码
    print("正在label编码")
    label = label_encode(train_aug_data)
    label.to_csv('./data/label.csv',index =False)
    train_sample.to_csv('./data/train_sample.csv',index =False)
    test_sample.to_csv('./data/test_sample.csv',index =False)
    print()


