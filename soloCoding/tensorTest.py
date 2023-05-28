import tensorflow as tf
import numpy as np
x_train = [1, 2, 3, 4, 5, 6, 7]
y_train = [1, 1, 1, 1, 1, 1, 1]

print("START")

# tf.random.uniform(shape=[arraySize, ...], minval=minvalueNumber, maxval=maxvalueNumber)
# [arraySize, ...] 공간 안에 minval 이상 maxval 이하의 랜덤값을 투입
W = tf.Variable(tf.random.uniform(shape=[1], minval=-100, maxval=100), name="weight")
b = tf.Variable(tf.random.uniform(shape=[1], minval=-100, maxval=100), name="bias")
 
hypothesis=x_train*W+b # 수식 세우기
 
#reduce_mean(value) : 배열 전체 원소들의 평균값
#reduce_mean(value, 0) : 행 단위로 평균을 낸다. reduce_mean([ [1 , 3] , [2 , 6]] , 1) == [(1 + 2) / 2 , (3 + 6) / 2] == [1.5 , 4.5]
#reduce_mean(value, 1) : 열 단위로 평균을 낸다. reduce_mean([ [1 , 3] , [2 , 6]] , 1) == [(1 + 3) / 2 , (2 + 6) / 2] == [2 , 4]
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # square == 제곱 | H(예측값) - Y(실제값) 의 제곱을 나타냄
sgd = tf.keras.optimizers.SGD(learning_rate=0.01) # 확률적 경사하강법. 한 번에 learning_rate만큼 점프함

model = tf.keras.models.Sequential() # 원하는 레이어를 순차적으로 추가하는 모델 Sequential을 사용(매우 직관적이다.)
model.add(tf.keras.layers.Dense(1, input_dim = 1)) # Sequential 모델을 생성
# tf.keras.layers.Dense(출력 뉴런 개수, input_dim = 입력 뉴런 개수)

# th.keras.layers.Dense 인자값
# units : 출력 값의 크기
# activation : 활성화 함수
#   - ‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
#   - ‘relu’ : rectifier 함수, 은익층에 주로 쓰입니다.
#   - ‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
#   - ‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.

# use_bias : 편향(b)을 사용할지 여부
# kernel_initializer : 가중치(W) 초기화 함수
# bias_iniotializer : 편향 초기화 함수
# kernel_regularizer : 가중치 정규화 방법
# bias_regularizer : 편향 정규화 방법
# activity_regularizer : 출력 값 정규화 방법
# kernel_constraint : 가중치에 적용되는 부가적인 제약 함수
# bias_constraint : 편향에 적용되는 부가적인 제약 함수

model.compile(loss='mean_squared_error',optimizer=sgd, metrics=[tf.keras.metrics.RootMeanSquaredError()]) # 손실함수로 mean_squared_error 를 , 옵티마이저로 sgd를 설정
# 옵티마이저 (Optimizer)는 손실 함수을 통해 얻은 손실값으로부터 모델을 업데이트하는 방식을 의미합니다.
# mean_squared_error : (예측값 - 실제값)²

model.fit(x_train,y_train,epochs=1000) # epochs 수 만큼 훈련을 진행한다.

# model.fit() 인자값 종류
# x=None,                     | 입력 데이터 (x값)
# y=None,                     | 대상 데이터 (y값)
# batch_size=None,            | 업데이트당 샘플 수
# epochs=1,                   | 훈련을 반복할 횟수
# verbose='auto',             | 실행 여부 ('0': 조용히 진행, '1': 표시줄 생성, '2': 훈련당 한 줄, 'auto': 보통 1이지만 ParameterServerStrategy와 사용하면 2)
# callbacks=None,             | 손실을 평가할 데이터와 각 에포크가 끝날 때의 모델 측정항목
# validation_split=0.0,       | 
# validation_data=None,       | 
# shuffle=True,               | 매 훈련마다 데이터를 셔플할 지에 대한 여부
# class_weight=None,          | 
# sample_weight=None,         | 
# initial_epoch=0,            | 훈련을 시작할 epoch(재교육에 사용)
# steps_per_epoch=None,       | 
# validation_steps=None,      | 
# validation_batch_size=None, | 
# validation_freq=1,          | 
# max_queue_size=10,          | 
# workers=1,                  | 
# use_multiprocessing=False   | 
 
print(model.predict(np.array([8]))) # 위 학습을 토대로 np.array[n]의 값을 '예측'하게 한다.