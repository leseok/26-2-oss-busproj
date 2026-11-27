# Ubuntu에서 scikit-learn 시계열 예제 재현

공식 예제 *Lagged features for time series forecasting*은 과거 자전거 대여량에서 특징을 만들고 다음 시간의 대여량을 예측한다. B 역할에서 이 예제를 선택한 이유는 예측 시점 이후의 정보가 학습이나 평가에 섞이지 않도록 시간순으로 분할하는 과정을 확인하기 위해서다. [공식 예제 설명](https://scikit-learn.org/1.9/auto_examples/applications/plot_time_series_lagged_features.html)

## 실제 실행 기록

Ubuntu 24.04.5 LTS ARM64 임시 가상 머신에서 공식 1.9.1 태그의 Python 파일 전체를 수정 없이 실행했다. 커널은 `6.8.0-151-generic`, Python은 3.12.3이다. scikit-learn 1.9.1, NumPy 2.5.5, SciPy 1.18.1, Polars 2.2.0, Matplotlib 3.11.2를 사용했다. 종료 코드는 0이었다.

```text
X shape: (17210, 13)
y shape: (17210,)
CV MAPE: 0.368 ± 0.072
```

과거 관측값으로 13개 특징을 구성한 뒤 남은 표본은 17,210건이었다. `TimeSeriesSplit`으로 세 번 평가했고, 각 분할의 학습 구간과 평가 구간 사이에 48개 관측을 비웠다. 0.368은 자전거 대여량의 평균 절대 백분율 오차를 세 분할에 걸쳐 평균한 값(36.8%)이고, 0.072는 분할 간 표준편차다. 예제의 일부 모델에는 난수 시드가 고정되어 있지 않아 재실행 시 수치가 달라질 수 있다.

6002번 만석 분류에 적용할 부분은 **예측 당시 알 수 있는 정보만 사용하고, 같은 운행을 한 묶음으로 유지하면서 이후 날짜에서 검증하는 방법**이다. 버스 기준선과 모델의 비교에는 A·B가 함께 확인한 목표 정류장 도착 좌석과 동일한 평가 사건이 필요하다.

## 재실행

Ubuntu의 새 작업 폴더에서 실행한다. 첫 실행에는 PyPI, GitHub, OpenML에 연결할 수 있어야 한다. Python 가상환경에만 패키지를 설치한다.

```bash
mkdir -p ~/bus6002-sklearn
cd ~/bus6002-sklearn
python3 -m venv --without-pip .venv
python3 - <<'PY'
from urllib.request import urlretrieve

urlretrieve("https://bootstrap.pypa.io/pip/pip.pyz", "pip.pyz")
source = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.1/examples/applications/plot_time_series_lagged_features.py"
urlretrieve(source, "plot_time_series_lagged_features.py")
PY
echo '49191414a48b76aa1ecea246d642ceb7e7a69dcda8197795ea0397c17707d04f  plot_time_series_lagged_features.py' | sha256sum -c - &&
.venv/bin/python pip.pyz install --only-binary=:all: scikit-learn==1.9.1 numpy==2.5.5 scipy==1.18.1 polars==2.2.0 matplotlib==3.11.2 &&
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 MPLBACKEND=Agg .venv/bin/python -u plot_time_series_lagged_features.py
```

예제 코드는 scikit-learn 개발진의 BSD-3-Clause 라이선스를 따른다. 입력은 OpenML의 *Bike Sharing Demand* 데이터셋 44063이다. 원본 스크립트가 Parquet 파일의 SHA-256을 확인한 뒤 읽는다. [1.9.1 예제 소스](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/examples/applications/plot_time_series_lagged_features.py) · [코드 라이선스](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/COPYING) · [OpenML 데이터셋](https://www.openml.org/d/44063)
