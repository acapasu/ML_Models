import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingClassifier


df_pen = sns.load_dataset('penguins')
df_pen = df_pen.dropna()
X = df_pen.drop(['species'], axis = 1)
y = df_pen['species']

categorical_x = ['island', 'sex']
numerical_x = X.drop(categorical_x, axis = 1).columns

categoricas = pd.get_dummies(X[categorical_x], drop_first=True)
X = pd.concat([categoricas, X[numerical_x]], axis = 1)

escalador = StandardScaler()
scaled_X = escalador.fit_transform(X)
model = GradientBoostingClassifier(max_depth = 5, n_estimators=10)
model.fit(scaled_X,y)

joblib.dump(model,'final_model.pkl')
joblib.dump(list(X.columns),'column_names.pkl')
joblib.dump(escalador,'escalador.pkl')
print('control')