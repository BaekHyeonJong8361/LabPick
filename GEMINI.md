# GEMINI.md — 학과서버 딥페이크 탐지 프로젝트 가이드

이 파일은 Gemini CLI가 이 프로젝트(`/home/t26106/deepfake`)에서 작업할 때 따라야 할 규칙·환경·컨텍스트를 정리한 가이드입니다.
**작업 시작 전 반드시 정독할 것.** 사용자(백현종, AI/딥러닝 담당)와 한국어로 협업.

---

## 0. 프로젝트 한 줄 요약

**목표:** SNS(인스타·구글 이미지·증명사진 등) 콘텐츠에서 딥페이크를 탐지하는 2-Stage Cascade 시스템 구축. 학술 데이터셋(FF++/DFDC/DF40)이 아니라 **실제 SNS 환경**이 최종 평가 기준이다.

**현재 가장 큰 걸림돌:** 도메인 시프트로 인한 SNS-real 이미지 false positive. 특히 Stage 1의 v7(Xception RGB)이 SNS 이미지를 매우 높은 확신으로 fake로 오탐. (`testreal{1,2,3}` 평가 결과 — §10 참조)

---

## 1. 학과서버 하드웨어

- **공유 서버** (다른 조도 사용 중일 가능성 있음). 항상 점유 확인 후 작업.
- GPU 2장 (RTX 4500 Ada 추정):
  - **GPU 0**: inference / 평가 / 가벼운 디버깅 전용
  - **GPU 1**: 학습 전용
- 동시 학습 절대 금지 (메모리 충돌). 새 학습 시작 전 항상 `nvidia-smi`로 GPU 1 비어있는지 확인.

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

---

## 2. Python 환경 (★ 가장 자주 실수하는 부분 ★)

학과서버에 두 python이 공존하며 **혼동하면 학습이 cuDNN 심볼 충돌로 죽음**:

| 경로 | torch | cuDNN | 학습 가능 |
|---|---|---|---|
| `/usr/bin/python` (bare `python`) | 2.2.2+cu121 | 8902 | ❌ backward 시 심볼 충돌 (`cublasSaxpy_v2` undefined) |
| `/home/t26106/.venv/bin/python` (activate 후) | 2.8.0+cu128 | 91002 | ✅ cuDNN ON 안전 |

**중요한 함정**: `.venv/bin/python`은 `/usr/bin/python`의 **심볼릭 링크**다. 절대경로로 그냥 호출하면 venv 격리가 안 되고 `~/.local/lib/python3.9/site-packages` (torch 2.2.2)가 shadow함. **반드시 `source activate` 해야 한다.**

### 학습 launch 정석 (반드시 이 형태)

```bash
nohup bash -c "source /home/t26106/.venv/bin/activate && python -u train_vN.py" > train_vN.log 2>&1 &
disown
```

또는

```bash
source /home/t26106/.venv/bin/activate
nohup python -u train_vN.py > train_vN.log 2>&1 &
disown
```

### 학습 시작 직후 검증

로그 첫 줄에 다음이 찍히는지 확인:
```
torch=2.8.0+cu128 | cudnn=91002 | enabled=True
```

만약 `torch=2.2.2`로 찍히면 venv 미적용 — **즉시 kill 후 재실행**.

### cuDNN 설정

- venv 환경에서: `cudnn.enabled = True` + `cudnn.benchmark = True` 유지 (학습 1.5~2배 빠름)
- system python에서: `cudnn.enabled = False` 필수 (CLAUDE.md §4 워크어라운드, 단 권장하지 않음 — venv로 가라)

---

## 3. 디렉토리 구조

```
/home/t26106/deepfake/
├── CLAUDE.md                      # Claude용 룰 (Gemini도 참고 권장)
├── GEMINI.md                      # 이 파일
├── dataset.py                     # 공통 dataset/augmentation
├── train_v6.py ~ train_v14.py     # 버전별 학습 스크립트 (★ 직접 수정 금지 ★)
├── inference_server_v3.py         # FastAPI 추론 서버 (포트 60007)
├── eval_sns_ood.py                # SNS OOD 평가 스크립트
├── run_xai.py                     # GradCAM/XAI
├── splits_v11_v2/                 # 학습/검증/테스트 split CSV
├── saved_models/
│   ├── xception_best_v7_combined.pth
│   ├── f3netlite_best_v8_combined.pth
│   ├── freq_only_best_v10.pth
│   ├── f3netlite_best_v11_df40.pth
│   ├── xception_best_v12_df40.pth
│   ├── freq_only_best_v13_df40.pth
│   ├── freq_only_best_v14_df40_aug.pth   # ← 현재 학습 중 (2026-05-11)
│   └── v{N}_results.json          # 각 버전 test 메트릭
├── test_inputs/
│   ├── testreal1.jpg              # ★ SNS OOD 평가셋 (§10) ★
│   ├── testreal2.png
│   └── testreal3.png
└── preprocessed_faces/            # FF++ / DFDC / DF40 얼굴 크롭 데이터
```

---

## 4. 코드 수정 규칙 (★ 매우 중요 ★)

### 4.1 절대 하지 말 것

- **기존 `train_v6.py` ~ `train_v14.py` 직접 수정 금지.**
  - 이유 1: 재현성·비교를 위해 버전별 코드 보존 필수
  - 이유 2: 코드 안전성 보호 메커니즘이 사용자 ML 코드 수정에 보수적으로 반응
- 새 실험은 **반드시 새 파일** (`train_v15.py`, `train_v16.py` ...)로 분리
- 공통 유틸은 import로 재사용:
  ```python
  from train_v6 import BinaryFocalLoss, get_eval_transforms
  from train_v8 import F3NetLite
  ```

### 4.2 dataset.py / 코어 모듈

- 신중히 수정. 새 augmentation은 가급적 학습 스크립트 안에 정의.
- 수정 전 사용자 확인 필수.

### 4.3 inference_server_v3.py

- 추론 서버는 사용자가 직접 운영 중. 임의로 수정 금지.
- 변경 필요 시 반드시 사전 동의.

---

## 5. 실험 설계: 한 번에 한 변수만 (★ v6~v14에서 검증된 원칙 ★)

- v6 → v7: 데이터만 변경 (FF++ → FF++/DFDC combined)
- v13 → v14: **augmentation만 변경** (다른 모든 것 동일)
- augmentation·loss·model·hyperparameter를 동시에 바꾸면 무엇이 효과를 냈는지 모름
- 새 실험 시작 전 명시: **"이번에 바꾸는 변수는 무엇이고, 통제되는 변수는 무엇인가?"**

---

## 6. 산출물 컨벤션

- 모델: `saved_models/{arch}_best_v{N}_{tag}.pth`
  - 예: `xception_best_v15_sns_aug.pth`
- 결과 JSON: `saved_models/v{N}_results.json` (test 메트릭 + 직전 버전 비교)
- 학습 로그: `train_v{N}.log`
- Split CSV: `splits_v{N}/` (버전별 분리해 재현성 확보)
- 리포트: `notion_v{N}_report.md`

---

## 7. 모델 아키텍처 요약

| 모델 | 아키텍처 | 입력 | 학습 데이터 | 역할 |
|---|---|---|---|---|
| v7  | timm xception, num_classes=1 | RGB | FF++/DFDC | Stage 1 |
| v8  | F3NetLite (RGB + Freq fusion) | RGB+Freq | FF++/DFDC | Stage 1 |
| v10 | FreqOnlyXception (FAD 9-band, no RGB) | Freq only | FF++/DFDC | Stage 1 |
| v11 | F3NetLiteV11 | RGB+Freq | DF40 | Stage 2 |
| v12 | timm xception | RGB | DF40 | Stage 2 |
| v13 | FreqOnlyXception | Freq only | DF40 | Stage 2 |
| v14 | FreqOnlyXception (= v13 + SNS-aug) | Freq only | DF40 | Stage 2 (실험) |

### FreqOnlyXception
- RGB를 보지 않고 FAD(Frequency-Aware Decomposition)로 9개 주파수 밴드만 분석
- `train_freq_only_v10.py` 에 정의
- DF40(SD2.1, DiT 등 최신 생성형) 의 generator fingerprint 잘 탐지

### Cascade ensemble (`inference_server_v3.py`)
- Stage 1 weights: `{v7: 0.3, v8: 1.0, v10: 1.0}` (v7 FPR 18.86%로 약함 → 0.3)
- Stage 2 weights: `{v11: 1.0, v12: 0.5, v13: 1.5}`
- 가중 logit-avg → sigmoid → prob
- Stage 1 threshold = 0.5, Stage 2 threshold = 0.55
- Stage 1이 FAKE면 즉시 반환, REAL이면 Stage 2로 escalate

---

## 8. GPU/프로세스 위생

### 학습 시작 전 체크
1. `nvidia-smi`로 GPU 1 비어있는지 확인
2. `pgrep -af train_v` 로 다른 학습 안 돌고 있는지 확인
3. 같은 이름의 `.log` 파일 있으면 백업 (`mv train_vN.log train_vN.log.bak`)

### 학습 중 모니터링
```bash
tail -f train_vN.log
nvidia-smi
```

### 학습 강제 종료 (사용자 사전 동의 필수)
```bash
pgrep -af train_vN | awk '{print $1}' | xargs -r kill -9
```

---

## 9. 자율 행동 vs 사용자 확인

### 자율적으로 진행해도 되는 것
- 명확한 에러 로그(Stack trace) 기반 수정
- 새 학습 스크립트 작성 (기존 코드 보존하며)
- 평가 스크립트 작성·실행
- 로그 분석·요약

### 반드시 사용자 사전 확인 필요
- 학습 프로세스 강제 종료 (kill)
- 디스크 다량 삭제
- 기존 모델/CSV 덮어쓰기
- `inference_server_v3.py` 수정
- 새 학습 시작 (어떤 변수를 바꾸는지 명시 후 OK 받기)

---

## 10. 표준 SNS OOD 평가셋 (★ 모든 모델 평가에 사용 ★)

3장 real 이미지 (모두 진짜 인물 사진):

| 파일 | 경로 | 종류 | 특징 |
|---|---|---|---|
| testreal1.jpg | `test_inputs/testreal1.jpg` | 본인(백현종) 증명사진 | 스튜디오 후처리, skin retouch, Samsung SM-N976N 메타 |
| testreal2.png | `test_inputs/testreal2.png` | 인스타그램 사진 캡처 | beauty filter 가능성, PNG |
| testreal3.png | `test_inputs/testreal3.png` | 김원호 선수 프로필 (구글) | 스튜디오 + CDN re-encoding |

**평가 기준:**
- 각 모델 prob ≤ 0.5 + cascade decision = REAL 이면 통과
- 새 모델 학습 후 반드시 이 3장으로 정성 평가
- 평가 스크립트: `python eval_sns_ood.py` (venv activate 후, GPU 0 권장)

**현재 (2026-05-11) baseline 상태:** 3장 모두 Stage 1에서 FAKE 판정 → 통과 X. v7이 0.84/0.97/0.97로 가장 큰 원흉.

---

## 11. 결과 보고 정직성

- 좋은 결과는 좋게, 나쁜 결과는 **숨기지 말고 정직하게** 말한다.
- v6 DFDC AUC 0.6860을 "낮다"고 정직하게 말한 게 v7 전략 결정에 도움 됐던 전례 있음.
- 수치 비교 시 **단위(AUC vs Accuracy)와 도메인(in-domain vs cross-domain)** 을 항상 명시.
- v_n vs v_{n-1} 직접 비교가 기본.

---

## 12. 한국어 응답 + 톤

- 사용자와 모든 대화는 **한국어** 로.
- 응답은 간결하게. 불필요한 사족·서두 생략.
- 핵심 결정 포인트는 명확히 묻기. 어중간한 추측 금지.
- 코드 경로·줄번호 인용 시 `file_path:line_number` 형식.

---

## 13. 자주 쓰는 명령 모음

```bash
# venv 활성화
source /home/t26106/.venv/bin/activate

# GPU 상태
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# 학습 launch (정석)
nohup bash -c "source /home/t26106/.venv/bin/activate && python -u train_vN.py" > train_vN.log 2>&1 &
disown

# 학습 진행 모니터
tail -f train_vN.log

# 학습 프로세스 확인
pgrep -af train_v

# OOD 평가
CUDA_VISIBLE_DEVICES=0 python eval_sns_ood.py

# 추론 서버 실행
/home/t26106/.venv/bin/python inference_server_v3.py
```

---

## 14. 현재 작업 상황 스냅샷 (2026-05-11)

- v14 학습 중 (GPU 1, epoch 3/25 진행, val AUC 1.0 saturate)
- v14 = v13 + SNS augmentation (bilateral / strong blur / median / double-JPEG / 강한 ColorJitter)
- 중간 평가 결과: v14가 v13 대비 testreal에서 오히려 prob 상승 (악화) — 학습 끝까지 가서 재평가 필요
- 다음 후보: Stage 1 retrain (v7 → v15, v8 → v16, v10 → v17)
- 더 자세한 분석은 사용자에게 직접 물어보거나 `MEMORY.md`(있을 경우) 참고

---

**요약:** venv activate, GPU 1=학습/GPU 0=추론, 새 버전은 새 파일, 한 번에 한 변수, testreal{1,2,3}로 정성 평가, 정직한 보고. 이게 핵심.
