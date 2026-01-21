# (기본 과제 1) Airflow를 활용한 Batch Serving

---

## 1. 과제 개요

본 과제는 **Apache Airflow**를 활용하여 머신러닝 모델의 Batch Serving 파이프라인을 구축하는 실습입니다. 총 4개의 DAG를 구현하며, 모델 학습, 모델 선택, 배치 추론, 외부 데이터 수집 등 실무에서 자주 사용되는 워크플로우를 경험합니다.

**연관 강의:** Product Serving - Batch Serving과 Airflow

---

## 2. 과제 출제 목적 및 배경

실제 ML 서비스 환경에서는 모델의 학습, 평가, 배포가 주기적으로 이루어져야 합니다. 이러한 반복적인 작업을 수동으로 처리하는 것은 비효율적이며, 휴먼 에러가 발생할 가능성이 높습니다.

Apache Airflow는 워크플로우 오케스트레이션 도구로서, 복잡한 데이터 파이프라인을 DAG(Directed Acyclic Graph) 형태로 정의하고 스케줄링할 수 있게 해줍니다. 이 과제를 통해 실무에서 널리 사용되는 Airflow의 기본 개념과 사용법을 익히고, Batch Serving 환경을 직접 구축해 봅니다.

---

## 3. 과제 수행으로 얻어갈 수 있는 역량

- **Airflow DAG 설계 및 구현 능력**: DAG 구조 설계, Task 간 의존성 정의, 스케줄링 설정
- **Operator 활용 능력**: PythonOperator, BranchPythonOperator, EmptyOperator 등 다양한 Operator 활용
- **모델 버저닝**: 학습된 모델을 버전별로 관리하고 저장하는 방법
- **조건부 분기 처리**: BranchPythonOperator를 활용한 조건부 워크플로우 구현
- **알림 시스템 구축**: Slack Webhook을 활용한 작업 성공/실패 알림 구현
- **외부 API 연동**: 기상청 API를 활용한 데이터 수집 파이프라인 구축

---

## 4. 과제 핵심 내용

### 구현할 DAG 목록 (총 4개)

| DAG       | 파일명                   | 설명                        | 스케줄     |
| --------- | ------------------------ | --------------------------- | ---------- |
| **DAG 1** | `01-simple-train.py`     | 주기적 모델 학습            | 매일 00:30 |
| **DAG 2** | `02-model-selection.py`  | 성능 기반 모델 선택 및 저장 | 매일 00:30 |
| **DAG 3** | `03-batch-inference.py`  | 배치 추론 (랜덤 5개 샘플)   | 5분마다    |
| **DAG 4** | `04-crawling_weather.py` | 서울 날씨 데이터 수집       | 1시간마다  |

### DAG 1: 주기적 모델 학습 (`01-simple-train.py`)

- Iris 데이터셋을 활용한 RandomForest 모델 학습
- 모델 버저닝을 통한 저장 (output 폴더)
- 학습 실패 시 Slack 알림 전송

### DAG 2: 모델 선택 (`02-model-selection.py`)

- BranchPythonOperator를 활용한 조건부 분기
- 기존 모델 대비 성능 향상 시에만 모델 업데이트
- 성공 시 Slack 알림 전송

### DAG 3: 배치 추론 (`03-batch-inference.py`)

- 저장된 모델 로드 및 추론
- 랜덤 5개 샘플에 대한 예측 수행
- 결과를 CSV 파일로 저장

### DAG 4: 날씨 데이터 수집 (`04-crawling_weather.py`)

- 기상청 초단기예보 API 호출
- 데이터 수집 → 전처리 → 저장 (2개 Task)
- 중복 데이터 제거 및 최신 데이터 유지

### 프로젝트 구조

```
dags/
├── 01-simple-train.py       # DAG 1: 모델 학습
├── 02-model-selection.py    # DAG 2: 모델 선택
├── 03-batch-inference.py    # DAG 3: 배치 추론
├── 04-crawling_weather.py   # DAG 4: 날씨 수집
└── utils/
    ├── __init__.py
    └── slack_notifier.py    # Slack 알림 유틸리티
```

---

## 5. Required Packages

### 패키지 설치

```bash
# Python 버전 확인
# IMPORTANT: 3.7.x ~ 3.11.x 로 설정되어야 합니다 (pyenv 사용 권장)
python --version

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 패키지 설치
pip3 install pip --upgrade

AIRFLOW_VERSION=2.6.3
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip3 install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# 외부 패키지 설치
pip3 install -r requirements.txt
```

### Airflow 초기화 및 실행

```bash
# 1. Airflow 초기화
export AIRFLOW_HOME=`pwd`
export TZ=Asia/Seoul
airflow db init

# 2. config 수정: airflow.cfg에서 load_examples = False로 설정

# 3. 사용자 생성
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# 4. airflow standalone 실행 (개발용)
airflow standalone

# 5. 접속: http://localhost:8080

# 6. 로그인: admin / admin
```

### Slack 연동 설정

1. Slack 개인 워크스페이스 생성
2. Incoming Webhook 앱 설정 및 Webhook URL 발급
3. Airflow Webserver에서 Connection 생성 (Conn ID: `my_webhook`)

### 기상청 API 설정

1. [공공데이터포털](https://www.data.go.kr) 회원가입
2. 단기예보 조회서비스 API 활용 신청
3. 발급받은 서비스 키를 `04-crawling_weather.py`에 설정

---

## 6. Reference

### 공식 문서

- [Airflow Operators](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/operators.html)
- [Custom Operator 만들기](https://airflow.apache.org/docs/apache-airflow/stable/howto/custom-operator.html)
- [BranchPythonOperator](https://airflow.apache.org/docs/apache-airflow/stable/howto/operator/python.html#branchpythonoperator)

### 외부 API

- [기상청 단기예보 조회서비스](https://www.data.go.kr/data/15084084/openapi.do)

---

WARNING: 본 교육 콘텐츠의 지식재산권은 재단법인 네이버커넥트에 귀속됩니다.
본 콘텐츠를 어떠한 경로로든 외부로 유출 및 수정하는 행위를 엄격히 금합니다.
다만, 비영리적 교육 및 연구활동에 한정되어 사용할 수 있으나 재단의 허락을 받아야 합니다.
이를 위반하는 경우, 관련 법률에 따라 책임을 질 수 있습니다.

