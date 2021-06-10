# https://github.com/tensorflow/docs-l10n/blob/master/site/ko/tutorials/keras/text_classification_with_hub.ipynb

import numpy as np

import tensorflow as tf

# !pip install tensorflow-hub
# !pip install tfds-nightly
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("허브 버전: ", hub.__version__)
print("GPU", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")


# 훈련 세트를 6대 4로 나눕니다.
# 결국 훈련에 15,000개 샘플, 검증에 10,000개 샘플, 테스트에 25,000개 샘플을 사용하게 됩니다.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# 데이터 탐색
# 잠시 데이터 형태를 알아 보죠. 
# 이 데이터셋의 샘플은 전처리된 정수 배열입니다. 
# 이 정수는 영화 리뷰에 나오는 단어를 나타냅니다. 
# 레이블(label)은 정수 0 또는 1입니다. 0은 부정적인 리뷰이고 1은 긍정적인 리뷰입니다.

# 처음 10개의 샘플을 출력해 보겠습니다.
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
# print("train_examples_batch:\n\n", train_examples_batch    )
# print("train_labels_batch:\n", train_labels_batch)


# 모델 구성
# 신경망은 층을 쌓아서 만듭니다. 여기에는 세 개의 중요한 구조적 결정이 필요합니다:

# 어떻게 텍스트를 표현할 것인가?
# 모델에서 얼마나 많은 층을 사용할 것인가?
# 각 층에서 얼마나 많은 은닉 유닛(hidden unit)을 사용할 것인가?
# 이 예제의 입력 데이터는 문장으로 구성됩니다. 예측할 레이블은 0 또는 1입니다.

# 텍스트를 표현하는 한 가지 방법은 문장을 임베딩(embedding) 벡터로 바꾸는 것입니다. 그러면 첫 번째 층으로 사전 훈련(pre-trained)된 텍스트 임베딩을 사용할 수 있습니다. 여기에는 두 가지 장점이 있습니다.

# 텍스트 전처리에 대해 신경 쓸 필요가 없습니다.
# 전이 학습의 장점을 이용합니다.
# 임베딩은 고정 크기이기 때문에 처리 과정이 단순해집니다.
# 이 예제는 텐서플로 허브에 있는 사전 훈련된 텍스트 임베딩 모델인 google/tf2-preview/gnews-swivel-20dim/1을 사용하겠습니다.

print("hub layer start \n ")

# 20차원 으로 훈련 된 데이터 
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
print(hub_layer(train_examples_batch[:3]))

# 모델 생성 
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

# 순서대로 층을 쌓아 분류기를 만듭니다:

# 첫 번째 층은 텐서플로 허브 층입니다. 이 층은 사전 훈련된 모델을 사용하여 하나의 문장을 임베딩 벡터로 매핑합니다. 여기서 사용하는 사전 훈련된 텍스트 임베딩 모델(google/tf2-preview/gnews-swivel-20dim/1)은 하나의 문장을 토큰(token)으로 나누고 각 토큰의 임베딩을 연결하여 반환합니다. 최종 차원은 (num_examples, embedding_dimension)입니다.
# 이 고정 크기의 출력 벡터는 16개의 은닉 유닛(hidden unit)을 가진 완전 연결 층(Dense)으로 주입됩니다.
# 마지막 층은 하나의 출력 노드를 가진 완전 연결 층입니다. sigmoid 활성화 함수를 사용하므로 확률 또는 신뢰도 수준을 표현하는 0~1 사이의 실수가 출력됩니다.

# 손실 함수와 옵티마이저
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 이 모델을 512개의 샘플로 이루어진 미니배치(mini-batch)에서 20번의 에포크(epoch) 동안 훈련합니다. 
# x_train과 y_train 텐서에 있는 모든 샘플에 대해 20번 반복한다는 뜻입니다. 
# 훈련하는 동안 10,000개의 검증 세트에서 모델의 손실과 정확도를 모니터링합니다:
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)              

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))                    