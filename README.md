# Boston Housing


## 구성 요소

```
data : 학습 원천 데이터 housing.csv 보관 폴더.
feature : 학습 데이터 numpy train_x, train_y, test_x, test_y 보관 폴더.
input : predict를 하기 위한 데이터 파일 보관 폴더.
model : 스케일러, 다항 생성기, 모델 등 보관 폴더.
output : predict을 완료 하면 결과 파일 보관 폴더

config.json : 환경설정 파일.

exploratory data analysis.ipynb : 각 데이터 속성별 탐색 수행
predictive modeling.ipynb : 예측 모델링 수행
preprocess.py : 학습을 위한 변환기 피팅, 전처리 및 피처 생성 수행
train.py : 모델링 결과가 적용된 예측 모델 학습 수행
predict.py : 학습된 모델 예측 동작 수행
```

## 파이썬 라이브러리

```
pip install -r requirements.txt
```

## 실행 방법

```
preprocess.py로 스케일러, 변환기 및 피처 생성 후 train.py 수행해야 하는 순서 필요
train.py 수행을 통해 모델 생성 후 predict.py이 되어야 함
```

### 1) preprocess.py

```
python preprocess.py

* 필수 조건 : 인터넷이 연결되어 있거나 data 폴더 내 housing.dat 필요
* 결과 출력 : model 폴더 내 MinMaxScaler, PolynomialFeatures 변환기 저장
            학습, 실험 데이터 분리하여 피처 저장
```

### 2) train.py

```
python train.py

* 필수 조건 : preprocess.py를 통해 생성되는 학습, 실험 데이터가 생성되어 있어야 함.
* 결과 출력 : model 폴더 내 예측기 저장
```

### 3) predict.py

```
python preprocess.py -i [input_file] -o [output_file]

* 필수 조건 : input_file은 input 폴더 내에 있어야 함
* 결과 출력 : 예측 수행 후 output 폴더에 출력 파일 저장
```