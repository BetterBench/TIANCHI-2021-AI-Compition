# 全球人工智能技术创新大赛 赛道一: 医学影像报告异常检测-不同方案的Baseline汇总

# 1 方案一(MachineLearning_model)
（1）环境
> python3.6

（2）得分:0.83+
（3）文件结构
```b
----utils.py  # 封装的数据处理函数
----extract_features.py  # 特征工程
----train_model.py  # 模型训练及生成提交文件
----adjust_parameter.py # 模型调参
```
（3）思路
传统的机器学习方法，最终使用XGBClassifier、LGBMClassifier、LogisticRegression三种模型融合，具体包含以下几个步骤，并附有博客详细介绍
+ [数据分析和探索](https://betterbench.blog.csdn.net/article/details/113858039)
+ [特征工程](https://betterbench.blog.csdn.net/article/details/113869071)
+ [模型训练]()
+ [模型调参]()

# 2 方案二(TextCNN-Tensorflow)待上传
（1）环境
> python3.6 
> Tensorflow 2.0
（2）得分：0.85+
（3）文件结构
