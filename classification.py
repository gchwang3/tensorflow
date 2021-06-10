# 출처 링크 : https://github.com/tensorflow/docs-l10n/blob/master/site/ko/tutorials/keras/classification.ipynb
# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 패션 MNIST 다운로드 
# 10개 범주 ( category) 70,000개 흑백 이미지 로 구성된 패션 mnist dataset 
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(type(train_images), type(train_labels))
print("train_labels:", train_labels[0])  # label 0 num 
# print("train_images:", train_images[0]) # image 0 28x28 array 

# traing image 정보 
print("train_images.shape", train_images.shape) # 이미지 갯수, 이미지 Width, Height 
print("len(train_labels)", len(train_labels)) # 이미지 갯수, 이미지 Width, Height 

print("test_images.shape", test_images.shape) # 이미지 갯수, 이미지 Width, Height 
print("len(test_labels)", len(test_labels)) # 이미지 갯수, 이미지 Width, Height 


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure()    # plt 형태 초기화 , 기본형 
# plt.imshow(train_images[0])
# plt.colorbar()  
# plt.grid(False) # 도표 안에 격자 유/무 
# plt.show()               

# 0~1 사이값으로 변경 
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10,10))   # UI 가로세로 사이즈 
# for i in range(25):
#     plt.subplot(5,5,i+1)    # 없을 경우 이미지 하나만 뜸.  rows, cols, 
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# 모델 구성 ( 층 설정 )
# 픽셀을 펼친 후에는 두 개의 tf.keras.layers.Dense 층이 연속되어 연결됩니다.
#  이 층을 밀집 연결(densely-connected) 또는 완전 연결(fully-connected) 층이라고 부릅니다.
#  첫 번째 Dense 층은 128개의 노드(또는 뉴런)를 가집니다. 
# 두 번째 (마지막) 층은 10개의 노드의 소프트맥스(softmax) 층입니다. 
# 이 층은 10개의 확률을 반환하고 반환된 값의 전체 합은 1입니다. 
# 각 노드는 현재 이미지가 10개 클래스 중 하나에 속할 확률을 출력합니다.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # 28x28 pixel to 1차원 배열로 변환 
    keras.layers.Dense(128, activation='relu'),     # dense 층  128 개 뉴런 또는 노드  
    keras.layers.Dense(10, activation='softmax')    # dense 층  10 개 노드 softmax 는 10개의 확률 을 리턴. 
])


# 모델 컴파일
# 모델을 훈련하기 전에 필요한 몇 가지 설정이 모델 컴파일 단계에서 추가됩니다:

# 옵티마이저(Optimizer)-데이터와 손실 함수를 바탕으로 모델의 업데이트 방법을 결정합니다.
# 손실 함수(Loss function)-훈련 하는 동안 모델의 오차를 측정합니다. 모델의 학습이 올바른 방향으로 향하도록 이 함수를 최소화해야 합니다.
# 지표(Metrics)-훈련 단계와 테스트 단계를 모니터링하기 위해 사용합니다. 다음 예에서는 올바르게 분류된 이미지의 비율인 정확도를 사용합니다.

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)           

# 정확도 평가
# 그다음 테스트 세트에서 모델의 성능을 비교합니다:

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\n테스트 정확도:', test_acc)


# 예측 만들기
# 훈련된 모델을 사용하여 이미지에 대한 예측을 만들 수 있습니다.

predictions = model.predict(test_images)

print(predictions[0])   # 첫번째 테스트 이미지 레이블별 예측값 
print(np.argmax(predictions[0])) # 에측값 배열에서 가장 높은값
print(test_labels[0])   # 레이블값  , 예측값과 일치함 확인. 

# test image 출력 및 예측값 정확도 x 축 label 에 출력
# 예측값 label 과  test_label 값이 동일하면 blue, 틀리면 red
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
# 예측값 label  class name, 정확도, 실제 label class name 
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


# 예측 값 bar 형식으로 출력 
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


# 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력합니다
# 올바른 예측은 파랑색으로 잘못된 예측은 빨강색으로 나타냅니다
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()


# 테스트 세트에서 이미지 하나를 선택합니다
img = test_images[0]

print(img.shape)    # (28, 28)

# 이미지 하나만 사용할 때도 배치에 추가합니다
img = (np.expand_dims(img,0))   # 차원 추가 

print(img.shape) # (1, 28, 28)

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])