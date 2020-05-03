#!/usr/bin/env python
# coding: utf-8

# In[45]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy import set_printoptions
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from pickle import dump
from pickle import load
import time, datetime
from xgboost import XGBClassifier
from xgboost import plot_importance
import xgboost as xgb


# # Get dummy variables if needed

# In[46]:


def make_dummy_columns(dfr, column):
    """gets dummy columns for variable

     Args:
     dfr: A dataframe
     column: column you want to break into dummies

     Returns:
     rearranged dataframe with new dummy columns for that variable."""
    df = pd.get_dummies(dfr, columns=[column])
    return df


# # Move predictor column to end

# In[47]:


# Move Y Column to End
def move_class_col(dfr, column_to_move):
    """moves class column to end.

     Args:
     dfr: A dataframe
     column_to_move: column you want to move to the end

     Returns:
     rearranged dataframe with column at end."""
    cols = list(dfr.columns.values)
    cols.pop(cols.index(column_to_move))
    dfr = dfr[cols + [column_to_move]]
    return dfr


# # Train test split and model run

# In[48]:


def run_model(model, train_x, train_y, test_x, test_y, X):
    print(model)
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    predicted = [round(value) for value in y_pred]
    matrix = confusion_matrix(test_y, predicted)
    print(matrix)
    print('precision: ', precision_score(test_y, predicted))
    print('recall: ', recall_score(test_y, predicted))
    print('roc_auc: ', roc_auc_score(test_y, predicted))
    print('accuracy: ', accuracy_score(test_y, predicted))
    print('f1_score:', f1_score(test_y, predicted))
    # (pd.Series(model.feature_importances_, index=X.columns[1:21])  #the feature count was hardcoded in


# .nlargest(5)
# .plot(kind='barh'))
# plot_importance(model)
# plt.show()


# # Run model and save it if needed

# In[86]:


df = pd.read_csv('iris.csv')
# df['Class-M/F']= df['Class-M/F'].map({Iris-virginica:0, Iris-setosa:1, Iris-versicolor:2})
df = df.loc[df['species'] != 'Iris-versicolor']
df['species'] = df['species'].map({'Iris-virginica': 0, 'Iris-setosa': 1})
df.species.value_counts()

# In[87]:


df.iloc[0:4]
{'sepal_length': 3, 'sepal_width': 2, 'petal_length': 3, 'petal_width': 6}
34
4.9
3.1
1.5
0.1

# In[88]:


df.shape
# df= pd.read_csv('/Users/jermainemarshall/Documents/intenders_conversion_prediction_no_w2v.csv')
# df= pd.read_csv('/Users/jermainemarshall/Documents/intenders_conversion_prediction_exclude_inside_pass_salesgrp.csv')
# df= pd.read_csv('/Users/jermainemarshall/Documents/intenders_conversion_prediction_kitchenaid_only.csv')


# In[89]:


if __name__ == '__main__':
    df = shuffle(df)
    dataset = df.values

    seed = 7

    # X = dataset[:,0:4]
    x = df[df.columns.difference(['species'])]
    y = df['species']
    # scaler = Normalizer().fit(X)
    # X = scaler.transform(X)
    # Y = dataset[:,4]
    # Y= Y.astype(int)
    # X = X.astype('float32')
    # split data into train and test sets
    # model will output the confusion matrix, precision, recall, roc_auc, and f1_curve. Will also print feature
    # importances
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.50, random_state=seed)
    # The below parameters were the best so far though it can vary 2-4 depending on seed%.
    xgb_model =  RandomForestClassifier() #LogisticRegression() #XGBClassifier(learning_rate=0.01, n_estimators=9, max_depth=5, subsample=0.99, colsample_bylevel=1.0,
                          #    gamma=3, scale_pos_weight=3, min_child_weight=5, seed=3)
    # run_model(xgb_model,X_train, Y_train, X_test, Y_test,df)
    xgb_model.fit(x, y)

# In[90]:


import seaborn as sns

corr = df.iloc[:, 0:4].corr()
# corr= corr.fillna(0)
# df_slice1= df_slice1.fillna(0)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, vmin=-1, vmax=1, linewidths=.5, center=0, ax=ax)

# In[91]:


filename = 'test_finalized_random_forest_iris_model.sav'
dump(xgb_model, open(filename, 'wb'))
# some time later...
# load the model from disk
# loaded_model = load(open(filename, 'rb'))


# In[92]:


from sklearn.externals import joblib

joblib.dump(xgb_model, 'test_finalized_random_forest_iris_model.pkl')

# In[93]:


cols_when_model_builds = list(df.columns[0:4])
joblib.dump(cols_when_model_builds, 'model_columns.pkl')
print("Models columns dumped!")
print(cols_when_model_builds)

# In[94]:


df.columns

# In[85]:


t = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
maxsum = -100
for i in range(len(t) + 1):
    summ = 0
    for j in range(i, len(t)):
        summ += t[j]
        if summ > maxsum:
            maxsum = summ
            start = i

            end = j
print(maxsum)

while start <= end:
    print(t[start])
    start += 1

# In[ ]:




