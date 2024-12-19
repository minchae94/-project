
### Pre-trained ResNet18을 이용한 MNIST image classification 모델



#### 개요

Transfer learning을 기반으로, MNIST dataset과 ResNet18 모델을 활용해 image classification 모델을 구현했다. Training은 두 단계로 진행했으며, 처음은 final layer, 두 번째는 layer 4를 파인튜닝한다. 이렇게 loss를 계산하고, 그 다음에 모델의 정확도를 평가한다.



#### 코드 구성

1. 데이터 로딩 및 전처리
* torch, torchvision, matplotlib 등의 라이브러리를 설치
* parameter 설정: batch size, learning rate, number of epoch을 설정
* mnist dataset을 resnet18 입력 형태로 맞추기 위해 transform을 정의(resize, normalize 등)
* load data: training할 데이터와 정확도 평가에 사용할 데이터를 각각 불러온다(train, test loader)
* pre trained ResNet18을 불러온다

2. Training
* loss function을 cross entropy loss로 설정, optimizer에 adam을 사용해 가중치 업데이트
* initial training: final layer만 training, learning rate 0.0001, epoch = 5
* additional training: layer4를 추가 training, learning rate 0.00001, epoch = 5
* back propagation 사용
* 각각의 loss를 epoch마다 출력

3. 평가
* test dataset을 활용해 정확도 계산
* 정확도 = 정확히 예측된 샘플 개수 / 전체 샘플 개수 * 100
