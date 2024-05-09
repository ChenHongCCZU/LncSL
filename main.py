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
train_label = [1] * len(train_p_data) + [0] * len(train_n_data)

# 读取 .pkl 文件来获取选定的特征索引
with open("/path/your/feature_lacation.pkl", "rb") as f:
    selected_by_at_least_four = pickle.load(f)
train_data = train_data[:, selected_by_at_least_four]

print("datas",train_data.shape)


test_data = np.concatenate([test_p_data, test_n_data], axis=0)
test_data = test_data[:, selected_by_at_least_four]
print("test_datas",test_data.shape)
test_label = [1] * len(test_p_data) + [0] * len(test_n_data)
train_label = np.array(train_label)
test_label = np.array(test_label)
print("labels",train_label.shape)
print("test_labels",test_label.shape)

# 模型定义
params = {
    "learning_rate": 0.01,
    "n_estimators": 5000,  # 设置一个大数以模拟许多迭代
    "max_depth": 8,
    "objective": "binary",
    "metric": "auc",
}
# rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.01, max_depth=8, loss='deviance')
xgb_model = XGBClassifier(learning_rate=0.01, n_estimators=3000, max_depth=8, objective='binary:logistic',n_jobs=-1, tree_method='auto')
lgbm_model = LGBMClassifier(**params, n_jobs=-1)
catboost_model = CatBoostClassifier(learning_rate=1e-2, iterations=3000, depth=8, loss_function="Logloss",eval_metric="AUC",
                                    verbose=500, task_type="CPU", devices='0:1')

# 训练模型池中的每个模型
model_pool = [gb_model, xgb_model, lgbm_model, catboost_model]


# 创建 StackingCVClassifier
stacking_classifier = StackingCVClassifier(
    classifiers=model_pool,
    meta_classifier=LogisticRegression(),
    use_probas=True,
    cv=5
)
# 训练 StackingCVClassifier
stacking_classifier.fit(train_data, train_label)


# 预测（用于测试集）
test_predictions_proba = stacking_classifier.predict_proba(test_data)[:, 1]

# 应用阈值
threshold = 0.5  # 你可以调整阈值
test_predictions = [0 if x < threshold else 1 for x in test_predictions_proba]

# 评估性能
tn, fp, fn, tp = metrics.confusion_matrix(y_true=test_label, y_pred=test_predictions).ravel()
recall = metrics.recall_score(y_pred=test_predictions, y_true=test_label)
precision = metrics.precision_score(y_pred=test_predictions, y_true=test_label)

se = tp / (tp + fn)
sp = tn / (tn + fp)

accuracy = metrics.accuracy_score(y_pred=test_predictions, y_true=test_label)
f1 = metrics.f1_score(y_pred=test_predictions, y_true=test_label)
mcc = metrics.matthews_corrcoef(y_pred=test_predictions, y_true=test_label)

auc = metrics.roc_auc_score(y_true=test_label, y_score=test_predictions_proba)
ap = metrics.average_precision_score(y_score=test_predictions_proba, y_true=test_label)

# 打印结果
print("tn", tn)
print("tp", tp)
print("fp", fp)
print("fn", fn)

print("recall", recall)
print("precision", precision)
print("se", se)
print("sp", sp)
print("accuracy", accuracy)
print("f1", f1)
print("mcc", mcc)
print("auc", auc)
print("ap", ap)
