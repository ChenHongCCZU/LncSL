from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import numpy as np
import sys, os,re,platform,math
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import orfipy_core
import Bio
from Bio import SeqIO
from Bio.Seq import Seq
import FrameKmer
from Bio.SeqUtils import ProtParam
import math
import random
import itertools
import re
from itertools import product
from collections import Counter
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from multiprocessing import Pool,cpu_count
pool = Pool(processes=cpu_count())
from sklearn.model_selection import train_test_split
print("load train")
train_p_data = np.load(r"/home/ys/work/ch/lncloc/lncloc/data/tiqu/S2_P_cksaap+ctdt+tpc.npy")
train_n_data = np.load(r"/home/ys/work/ch/lncloc/lncloc/data/tiqu/S2_N_cksaap+ctdt+tpc.npy")
train_p_data, test_p_data = train_test_split(train_p_data, test_size=0.2)

train_n_data, test_n_data = train_test_split(train_n_data, test_size=0.2)
train_data = np.concatenate([train_p_data, train_n_data], axis=0)
test_data = np.concatenate([test_p_data, test_n_data], axis=0)
train_label = [1] * len(train_p_data) + [0] * len(train_n_data)
test_label = [1] * len(test_p_data) + [0] * len(test_n_data)
train_label = np.array(train_label)
test_label = np.array(test_label)


base_classifier = DecisionTreeClassifier(max_depth=8)
params = {
    "learning_rate": 0.01,
    "n_estimators": 15000,  # 设置一个大数以模拟许多迭代
    "max_depth": 8,
    "objective": "binary",
    "metric": "auc",
}
# 创建一个模型池，包括多个不同的分类器
model_pool = [
    RandomForestClassifier(n_estimators=50, random_state=42),
    AdaBoostClassifier(base_classifier, n_estimators=500, learning_rate=1e-2),
    GradientBoostingClassifier(n_estimators=15000, learning_rate=0.01, max_depth=8, loss='deviance', random_state=42),
    ExtraTreesClassifier(n_estimators=15000, max_depth=8, criterion="entropy", max_features="auto", bootstrap=False,
                         random_state=42, verbose=500),
    LogisticRegression(),
    SVC(probability=True),
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    XGBClassifier(learning_rate=0.01, n_estimators=15000, max_depth=8, objective='binary:logistic',
                  n_jobs=-1, tree_method='auto'),
    LGBMClassifier(**params, n_jobs=-1),
    CatBoostClassifier(learning_rate=1e-2, iterations=15000, depth=8, loss_function="Logloss",
                       verbose=500, task_type="CPU", devices='0:1')
]

# 训练模型池中的每个模型
model_outputs = np.zeros((len(test_data), len(model_pool)))

for i, model in enumerate(model_pool):
    # 训练模型
    model.fit(train_data, train_label)

    # 记录每个模型的输出概率
    model_outputs[:, i] = model.predict_proba(test_data)[:, 1]

# 使用另一个随机森林进行特征选择
feature_selector = SelectFromModel(RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5))
feature_selector.fit(model_outputs, test_label)
selected_features = feature_selector.get_support()

# 打印选择的特征及其对应的模型索引
for i, selected in enumerate(selected_features):
    if selected:
        print(f"Selected Feature {i} is from Model {i}")