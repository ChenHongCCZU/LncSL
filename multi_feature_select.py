import numpy as np
import sys, os,re,platform,math
from sklearn import metrics
import pandas as pd
import pickle
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
print("Begin")
train_data = np.load(r"/your/path/feature.npy")
train_label = np.load(r"/your/path/label.npy")
# 1. 方差阈值方法
variance_selector = VarianceThreshold(threshold=1.5e-6)
variance_selector.fit(train_data)
selected_columns_variance = np.where(variance_selector.get_support())[0]


# 2. 单变量特征选择
k_best_selector = SelectKBest(score_func=f_classif, k=250)
k_best_selector.fit(train_data, train_label)
selected_columns_k_best = np.where(k_best_selector.get_support())[0]

# 3. 递归特征消除
model = RandomForestClassifier()
rfe_selector = RFE(model, n_features_to_select=300)
rfe_selector.fit(train_data, train_label)
selected_columns_rfe = np.where(rfe_selector.support_)[0]

# 4. 基于树模型的特征选择
tree_model = RandomForestClassifier()
tree_model.fit(train_data, train_label)
feature_importances = tree_model.feature_importances_
tree_selector = feature_importances > 1.5e-6  # 以示例值为例，根据实际情况调整
selected_columns_tree = np.where(tree_selector)[0]


# 5. L1正则化（LASSO）
from sklearn.linear_model import LogisticRegression

lasso_model = LogisticRegression(penalty='l1', solver='liblinear')
lasso_model.fit(train_data, train_label)
lasso_selector = lasso_model.coef_.flatten() != 0
selected_columns_lasso = np.where(lasso_selector)[0]


# 6. 互信息
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(train_data, train_label)
mi_selector = mi_scores > 0.01  # 以示例值为例，根据实际情况调整
selected_columns_mi = np.where(mi_selector)[0]


# 7. Boruta
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
boruta_selector.fit(train_data, train_label)
selected_columns_boruta = np.where(boruta_selector.support_)[0]


# 假设 train_data 的形状是 [n_samples, n_features]
n_features = train_data.shape[1]

# 初始化一个计数器数组，长度等于特征数量
feature_counts = np.zeros(n_features)

# 定义一个添加选择特征索引的函数
def add_selected_features(selected_features):
    for idx in selected_features:
        feature_counts[idx] += 1

# 对每种方法选出的特征进行计数
add_selected_features(selected_columns_variance)
add_selected_features(selected_columns_k_best)
add_selected_features(selected_columns_rfe)
add_selected_features(selected_columns_tree)
add_selected_features(selected_columns_lasso)
add_selected_features(selected_columns_mi)
add_selected_features(selected_columns_boruta)

# 选择至少被四种方法选中的特征
selected_by_at_least_four = np.where(feature_counts >= 4)[0]

import pickle
with open("/your/path/feature_lacation.pkl", "wb") as f:
    pickle.dump(selected_by_at_least_four, f)