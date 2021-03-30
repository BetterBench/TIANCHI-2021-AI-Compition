import numpy as np
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import random
# 计算mloloss
def Mutilogloss(actual, predict, eps=1e-5):
    allloss = []
    for i in range(actual.shape[1]):
        loss = log_loss(actual[:,i],predict[:,i])
        allloss.append(loss)
    return np.sum(allloss)/actual.shape[1]

# 统计每行的句子长度并作为句子的一个特征
def count_text_len(data):
    text_len =[]
    datalen = len(data)
    for i in range(0,datalen):
        one_lines = ''.join(list(data['text'][i]))
        len_text = one_lines.strip().split(" ")
        text_len.append(len(len_text))
    return text_len
# 删除句子中高频词
def delete_highfrequency_word(data):
    text_Word_frequency = []
    from collections import Counter
    datalen = len(data)
    for i in range(0,datalen):
        one_lines = ''.join(list(data['text'][i][1:-1]))
        textls = one_lines.strip().split(" ")
        all_word_count = Counter(textls)
        all_word_count = sorted(all_word_count.items(), key=lambda d:d[1], reverse = True)
        # 删除693和328两个字符，根据数据分析阶段的词频统计，这两个字符可能是标点符号
        dict_word_count = dict(all_word_count)
        if dict_word_count.get('693') !=None:
            del dict_word_count['693']
        if dict_word_count.get('328') !=None:
            del dict_word_count['328']
        string_top10_high_frequency_word = list(dict_word_count.keys())
        if '' in string_top10_high_frequency_word:
            string_top10_high_frequency_word.remove('')
        list_to_str  = " ".join(string_top10_high_frequency_word)
        text_Word_frequency.append(list_to_str)
    return text_Word_frequency

# 缺失值填充，空label的填充去17
def label_fill(data):
    new_code_label =[]
    datalen = len(data)
    for i in range(0,datalen):
        one_lines = ''.join(list(data['label'][i][1:])).strip()
        if one_lines =='':#空label
            new_code_label.append('17')
        else:
            new_code_label.append(one_lines)

    return new_code_label

# label编码，一个编码占一个表格，编码后一个标签有18列
def label_encode(train_data):
    train_data_label = pd.DataFrame(columns=['label'])
    train_data_label['label'] = label_fill(train_data)
    mlb = MultiLabelBinarizer(classes=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17'])#17表示无异常
    Ylist = []
    all_train_data_label = pd.DataFrame(columns=['f0','f1', 'f2', 'f3','f4','f5','f6','f7', 'f8', 'f9','f10','f11','f12','f13', 'f14', 'f15','f16','f17'],index=[])
    indexsize = 0
    for i in range(len(train_data_label)):
        templist = train_data_label['label'][i].split()
        # 转二值化编码
        label_code_list = list(mlb.fit_transform([templist])[0])
        # 写入DataFrame
        all_train_data_label.loc[indexsize] =label_code_list
        indexsize = indexsize + 1 
    return all_train_data_label
    
