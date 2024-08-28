# BehaPulse-Sensing-Model

## 목차
- [개요](#개요)
- [주요기능](#주요기능)
- [설치 방법](#설치방법)
- [사용법](#사용법)
- [디렉토리구조](#디렉토리구조)
- [라이선스](라이선스)

## 개요

BehaPulse-Sensing-Model은 저가형 ESP32 장치의 Channel State Information (CSI) 데이터를 활용하여 일상 생활 행동을 감지하는 딥러닝 기반 프로젝트입니다. 이 프로젝트는 일상적인 환경에서 인간의 행동을 모니터링하고 분석하는 데 중점을 두고 있으며, 저렴한 하드웨어를 사용하여 접근성을 높였습니다.

## 주요 기능

- **저비용 하드웨어**: ESP32 기반의 저렴한 장치를 사용하여 데이터 수집.
- **딥러닝 모델**: 수집된 CSI 데이터를 처리하고 행동을 예측하는 딥러닝 모델 제공.
- **모델 학습 및 추론**: 사용자 정의 가능한 모델 학습 및 추론 스크립트 포함.

## 설치방법

1. **저장소 클론**:
    
    ```bash
    git clone <https://github.com/BehaPulse/BehaPulse-Sensing-Model.git>
    cd BehaPulse-Sensing-Model
    ```
    
2. **의존성 설치**:
    
    ```bash
    pip install -r requirements.txt
    ```

## 사용법
### 모델
1. **모델 학습**:
    
    ```bash
    python train.py
    ```
    
    먼저, `config/train_config.yaml` 파일을 열어 학습 설정을 수정해야 합니다. 이 파일에서 데이터셋 경로, 모델 하이퍼파라미터, 학습률, 배치 크기 등 중요한 매개변수를 설정할 수 있습니다. 각 설정은 프로젝트의 요구 사항에 맞게 조정해야 합니다. 설정 파일을 수정한 후, 위 명령어를 실행하여 학습을 시작할 수 있습니다.
    
2. **모델 추론**:
    
    ```bash
    python inference.py
    ```
    
    추론을 실행하기 전에 `config/inference_config.yaml` 파일을 열어 설정을 조정해야 합니다. 이 파일에서 모델 체크포인트 경로, 입력 데이터 경로, 배치 크기 등의 매개변수를 설정할 수 있습니다. 설정 파일을 적절히 수정한 후, 위 명령어를 실행하여 추론을 시작합니다.

### 데이터 준비

모델 학습에 사용할 데이터를 준비하려면, `data/` 디렉토리에 데이터를 저장합니다. 이 데이터는 `config/train_config.yaml` 파일에서 지정된 경로와 일치해야 합니다. 데이터는 CSV 형식으로 준비하며, 각 열은 다양한 특성을 나타냅니다.

### 결과 분석

모델 학습 및 추론 결과는 `results/` 디렉토리에 저장됩니다. 이 디렉토리에서 모델의 예측 결과를 확인할 수 있으며, 추가적인 분석을 수행할 수 있습니다.

## 디렉토리구조

```
BehaPulse-Sensing-Model/
        ├── README.md
        ├── config # 설정 디렉토리
        │   ├── inference_config.yaml
        │   └── train_config.yaml
        ├── data # 데이터 디렉토리
        ├── inference.py # 추론
        ├── models # 모델 디렉토리
        │   ├── ViT.py
        │   └── utils.py
        ├── modules # 모듈 디렉토리
        │   ├── datasets.py
        │   ├── earlystoppers.py
        │   ├── losses.py
        │   ├── metrics.py
        │   ├── optimizers.py
        │   ├── recorders.py
        │   ├── trainer.py
        │   └── utils.py
        ├── requirements.txt
        ├── results # 결과 저장 디렉토리
        │   └── train
        └── train.py # 학습
```
## 라이선스

이 프로젝트는 MIT 라이선스에 따라 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하십시오.

