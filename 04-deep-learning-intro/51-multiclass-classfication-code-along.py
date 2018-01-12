import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# seaborn: statistical data visualization
# https://seaborn.pydata.org/

# panda 이용해 iris.csv 읽어 온다
# panda는 데이터 분석 라이브러리
df = pd.read_csv('data/iris.csv')

# seebon으로 데이터 상관 관계를 시각화 한다
# seaborn은 통계 데이터 시각화 라이브러리
# sns.pairplot(df, hue="species")

# 플로팅 결과를 보여준다
# plt.show()

# 데이터의 'species' 컬럼을 삭제한다.
# axis = 1 : 컬럼의 레이블을 삭제한다.
X = df.drop('species', axis=1)

# 유니크한 'species' 값을 만든다
target_names = df['species'].unique()

# dict 타입으로 만든다.
# {'setosa': 0, 'versicolor': 1, 'virginica': 2}
target_dict = {n:i for i, n in enumerate(target_names)}

# 'setosa' 0, 'versicolor' 1, 'virginica' 3
# series 데이터로 만든다
y = df['species'].map(target_dict)

# 'seotsa' [ 1.  0.  0.], 'versicolor' [ 0.  1.  0.], 'virginica' [ 0.  0.  1.]
# 이진 클래스 행렬로 만든다.
y_cat = to_categorical(y)

# 데이터를 트레이닝 셋과 테스트 셋으로 나눈다
# X.values 데이터, y_cat 결과 레이블
# test_size 20%
# X_Train 트레이닝 데이터 셋, y_train 트레이닝 데이터 셋 결과 레이블
# X_test 테스트 데이터 셋, y_test 테스트 데이터 셋 결과 레이블
X_Train, X_test, y_train, y_test = train_test_split(X.values, y_cat, test_size=0.2)

# Keras Sequential 모델을 만든다
model = Sequential()

# Dense를 구성해 model에 추가한다
# 3 : 출력층, 'seotosa', 'versicolor', 'virginica' 중 하나
# input_sape=(4,) 입력층 4개의 iris 데이터가 입력층으로 들어간다.
# activation='softmax' 활성화 함수로 softmax 사용
model.add(Dense(3, input_shape=(4,), activation='softmax'))

# 모델 트레이닝을 위한 컴파일 프로세스를 구성한다.
# optimizer: Adma / leraning rate = 0.1
# loss: loss function / categorical_crossentropy
# matrics: 측정할 항목. 분류 문제일 경우 보통 accuracy 측정
model.compile(Adam(lr=0.1), loss="categorical_crossentropy", metrics=['accuracy'])

# 모델을 트레이닝 한다
# validation_split: 유효성 검사 할 데이터 비율
model.fit(X_Train, y_train, epochs=20, validation_split=0.1)

# 테스트 데이터로 예측 한다
# 'seotosa', 'versicolor', 'virginica' 예측 수치로 나온다.
# [  9.54003513e-01   4.59017791e-02   9.47759836e-05] ...
y_pred = model.predict(X_test)


# axis에 가장 큰 값을 갖는 인덱스를 찾는다.
# y_test 즉, X_Test의 결과 레이블 값 
# y_pred 즉, X_Test로 예측한 결과를 
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)


# 레포트
print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))

