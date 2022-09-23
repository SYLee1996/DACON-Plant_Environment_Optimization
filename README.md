# DACON-Plant_Environment_Optimization(생육 환경 최적화)

## RNN & SENet Regressor + Shake-ResNet Ensemble

+ 주최 및 주관: 데이콘 
+ 링크: https://dacon.io/competitions/official/235900/overview/description

+ 청경채 사진과 환경 데이터를 활용한 잎면적 예측 알고리즘 개발

----
## Summary
+ ### Data Processing

    + Clustering
      + Clustering 기법 중 FeatureAgglomeration 방법을 통해 청경채 잎만을 군집화
          + 군집화 후 배경 부분 마스킹 -> 군집화된 청경채 이미지와 raw image를 합침
          
          ![image](https://user-images.githubusercontent.com/30611947/191727281-25a056f2-9dab-49ce-ab6e-125bde3420b3.png)

          
          + 군집화된 청경채 이미지 픽셀값의 합을 통해 경향 파악
          

          ![image](https://user-images.githubusercontent.com/30611947/191729693-c081c61b-637f-49af-be37-050891bbfc2d.png)

  </br>

    + Augmentation
      1. Shake-Shake regularization 이용
          + ResNet을 이용하여 Internal Representations에 augmentation 적용
          
          ![image](https://user-images.githubusercontent.com/30611947/191730879-1b2eaee1-5b00-4472-a0c0-23cb379b0a06.png)

      
      2. Filter를 이용한 edge detect
          + Sobel filter를 이용한 증강을 추가(다양한 filter 기반 edge detect 하여 unsharp image와 합친 이미지를 만들어 사용)
          + 여러 filter 및 LBP를 이용한 augmentation을 시도, 결과적으로 Sobel filter를 이용한 경우 가장 높은 성능 보임
          
          ![image](https://user-images.githubusercontent.com/30611947/191729969-0293b0a2-4849-4e2b-8b5c-0e24459f3b55.png)
          
  </br>

    + 환경데이터 전처리
      + 분포 확인 후 이상치 처리 및 보간 후 정규화(min-max scaling) 수행
      + 군집화된 청경채 이미지 픽셀값의 합을 새로운 변수로 생성
      
----
  
+ ### Model
    + 환경 데이터: RNN 및 SENet을 이용하여 임베딩

    + 이미지 데이터: Pre-trained imageNet model 및 Shake-ResNet을 이용하여 임베딩

    + Multi-modality data를 concat 후 FC 레이어를 통한 classification

  </br>

    + Model techniques
      + scheduler: CosineAnnealingLR
      + Loss : L1-loss 사용(Focal 및 SmoothCrossEntropyLoss 사용 시 성능 변동 없음)
      + optimizer : AdamW 사용
      + EarlyStopping 사용
      + automatic mixed precision 사용
      + 5-Fold Cross validation 수행

----

+ ## Environment 
  + 사용한 Docker image는 Docker Hub에 첨부하며 cuda10.2, cudnn7, ubuntu18.04 환경을 제공합니다.
    + https://hub.docker.com/r/lsy2026/plant_diseases
  
  
+ ## Libraries
  + python==3.9.7
  + pandas==1.3.4
  + numpy==1.20.3
  + tqdm==4.62.3
  + sklearn==0.24.2
  + cv2==4.5.5
  + albumentations==1.1.0
  + torch==1.11.2+cu102
  + torchvision==0.12.0+cu102
  + timm==0.6.7

---- 

+ ## 개선할 점
  
  + xgboost, lightgbm와 같은 머신러닝 기법을 이용하는 방식 고려
  + 클러스터링 기법 뿐만이 아닌 BGR 이미지를 HSV로 변환하여 마스킹 하는 방식 고려(이미지 밝기 조절 -> HSV로 색상 object 마스킹-> 마스킹 된 이미지 픽셀 비율 추출)
  + X는 이미지에서 추출한 잎의 픽셀수, y는 라벨(leaf_weight : 해당 이미지가 촬영된 시점으로부터 1일 후의 잎 면적)의 예측 진행방식 고려
  + 이미지 데이터만으로 예측하고, 환경 데이터만으로 예측하여 앙상블 방법 고려(CatBoost: 이미지 픽셀 비율 Feature 변수로 포함 -> 메타데이터 전처리 (결측값/변환/이상값) -> 메타데이터로만 CatBoost 학습 -> Test 데이터셋 예측)
  + 환경 데이터의 머신러닝 모델 구축 시, Feature간 상관관계가 높은 변수는 제거하기(총추정광량은 백색광추정광량, 적색광추정광량, 청색광추정광량 합이므로 총추정광량 변수 제거하기)

  + 이미지 데이터 전처리 시, 군집화된 청경채 이미지 픽셀값의 합을 통해 경향 파악하여 이상 데이터 제외하는 방식 고려
