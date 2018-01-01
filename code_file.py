import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import to_categorical
from numpy import argmax
from sklearn.preprocessing import StandardScaler






df = pd.read_csv('survey_cleaned.csv')
X = df.drop('treatment',axis = 1)
y = df['treatment']

column_names = ["Gender","self_employed","family_history","no_employees","remote_work","tech_company","benefits","care_options","wellness_program","seek_help","anonymity","leave","mental_health_consequence","phys_health_consequence","coworkers","supervisor","mental_health_interview","phys_health_interview","mental_vs_physical","obs_consequence"]
df_categorized = df[['Age']]
for element in column_names :
    df_temp = pd.get_dummies(df[element],prefix = element, prefix_sep = '_')
    df_categorized = df_categorized.join(df_temp)


X = df_categorized.values

y = pd.get_dummies(df['treatment'],prefix = 'treatment',prefix_sep = '_')
y = y.values

scaler = StandardScaler()
y = scaler.fit_transform(y)



print(y)
n_cols = X.shape[1]
model = Sequential()
model.add(Dense(100,activation = 'relu',input_shape = (n_cols,)))
model.add(Dense(500,activation = 'relu',input_shape = (n_cols,)))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X,y,epochs = 20)
