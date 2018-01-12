"""
pima indeans diabetes
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# panda로 데이터 읽기
# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
# 6,148,72,35,0,33.6,0.627,50,1
# 1,85,66,29,0,26.6,0.351,31,0
# ...

df = pd.read_csv('data/diabetes.csv')
# print(df.info())
# print(df.describe())
# sns.pairplot(df, hue="Outcome")
# sns.heatmap(df.corr(), annot = True)
# plt.show()

# 데이터의 'Outcome' 컬럼 삭제하고
# StandardScaler를 사용해 데이터 스케일을 조절한다.
sc = StandardScaler()
X = sc.fit_transform(df.drop('Outcome', axis=1))
Y = to_categorical(df['Outcome'].values) # [0, 1] or [1, 0]

# 데이터를 트레이닝 셋과 테스트 셋으로 나눈다.
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, random_state=22, test_size=0.2)

model = Sequential()
model.add(Dense(32, input_shape=(8,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(Adam(lr=0.05), loss="categorical_crossentropy", metrics=['accuracy'])

model.summary()
model.fit(X_train, Y_train, epochs=20, validation_split=0.1)

Y_pred = model.predict(X_test)

Y_test_class = np.argmax(Y_test, axis=1)
Y_pred_class = np.argmax(Y_pred, axis=1)

# print(Y_test_class)
# print(Y_pred_class)

print("\naccuracy_score\n", accuracy_score(Y_test_class, Y_pred_class))
print("\nclassification_report\n", classification_report(Y_test_class, Y_pred_class))
print("\nconfusion_matrix\n", confusion_matrix(Y_test_class, Y_pred_class))