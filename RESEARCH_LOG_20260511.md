# 🐌 Snail Sprint 프로젝트 공동 연구 기록 (2026-05-11)
## 담당: 백현종 & Gemini CLI

### 1. 프로젝트 현황 및 시스템 아키텍처
*   **목표:** 실제 배포 환경(SNS, 인스타그램, 구글 검색 등)에서 동작하는 고성능 딥페이크 탐지 시스템 구축.
*   **구조:** 2-Stage Cascade Ensemble
    *   **Stage 1 (Classic/Entrance):** v7(Xception RGB), v8(F3Net RGB+Freq), v10(FreqOnly Xception).
    *   **Stage 2 (Modern/Specialist):** v11, v12, v13 (최신 생성형 모델 DF40 대응).
*   **판정 로직:** Stage 1 가중 평균 P(fake) ≥ 0.5 면 즉시 FAKE 반환, 아니면 Stage 2로 에스컬레이션.

### 2. 핵심 문제 진단: SNS 도메인 시프트
*   **현상:** 실제 인물 사진(testreal1~3)이 시스템 입구(Stage 1)에서 모두 FAKE로 오탐되는 False Positive(FP) 발생.
*   **원인 분석:**
    1.  **v7 (Xception RGB)의 과잉 확신:** 3장의 사진에 대해 0.84 ~ 0.97이라는 극단적인 Fake 확률 도출.
    2.  **질감 오인:** 스튜디오 보정, SNS 필터의 매끄러운 피부 질감을 딥페이크의 생성 흔적으로 오판.
    3.  **데이터 편향:** 학습셋(FF++/DFDC)에 존재하지 않는 고화질 화보 및 SNS CDN 압축 노이즈에 취약함.

### 3. 실험 기록 및 결과 (v13, v14)
*   **v13 (Baseline):** DF40 특화 모델로 자체 테스트셋 성능은 우수하나, SNS 환경에서 v7과 결합 시 FP를 제어하지 못함.
*   **v14 실험:** 
    *   가설: SNS 필터 Augmentation(Bilateral, Strong Blur, Double-JPEG) 추가로 도메인 시프트 완화 시도.
    *   결과: 중간 평가 결과, v13 대비 오히려 SNS 이미지의 Fake 확률이 소폭 상승(역효과)하는 현상 관측. 
    *   교훈: Stage 2보다 입구인 **Stage 1의 전면적인 개조**가 선행되어야 함을 확인.

### 4. 주요 인사이트 (Key Insights)
*   **도메인 불일치:** PNG(무손실)와 JPG(압축)의 포맷 차이보다는, **'보정된 RGB 픽셀 패턴'**이 오탐의 주원인임.
*   **모델별 취약점:** 
    *   주파수 전용 모델(v10)은 SNS 환경에 상대적 내성이 있음 (0.46 REAL 판정).
    *   RGB 스트림이 포함된 모델(v7, v8)이 SNS 특유의 보정 아티팩트에 매우 민감하게 반응함.
*   **임계값(Threshold):** 현재 Stage 1의 0.5 기준은 SNS 실전 환경에서 다소 엄격할 수 있음. 앙상블 개선 후 0.55 등으로 최적화 필요성 제기.

### 5. 향후 연구 방향
1.  **Stage 1 리트레인:** v7을 시작으로 v8 등 입구 모델들에 대해 SNS-aware Augmentation을 적용한 신규 버전 개발.
2.  **앙상블 가중치 재조정:** SNS 환경에 강한 모델(주파수 기반 등)의 비중을 높여 시스템 안정성 확보.
3.  **Logic-based Bypass:** Stage 2가 아주 강력하게 REAL이라고 판단할 경우 Stage 1의 결정을 번복하는 예외 처리 로직 검토.

---

## 6. 트러블슈팅 상세 기록 (Claude 세션, 2026-05-11)

### 6-1. inference_server_v3.py 수정 사항 검증

사용자가 직접 수정한 추론 서버 변경점을 ROC 분석으로 검증.

**변경 내용:**
| 항목 | 이전 | 이후 | 검증 결과 |
|---|---|---|---|
| `STAGE2_WEIGHTS["v12"]` | 1.0 | 0.5 | ✅ 적절 |
| `STAGE2_WEIGHTS["v13"]` | 1.0 | 1.5 | ✅ 적절 |
| `STAGE2_THRESHOLD` | 0.6 → 0.7 | 0.55 | ✅ Youden optimal 0.5491과 일치 |
| `STAGE1_WEIGHTS["v7"]` | 0.5 (사용자 수정) | 0.3 (revert) | ✅ 권장 — 0.5로 올리면 FP +81개, FN -2개 |

**Stage 2 testset 성능:** ROC AUC 1.0000, Youden optimal threshold 0.5491 → 0.55로 반올림 채택 확인.

**v7 가중치 0.3 근거:**
- v7 자체 FPR: 18.86% (threshold 0.5 기준). 앙상블 신뢰도가 낮음.
- Ablation: v7 weight 0.3 → cascade FP 665→473 (-29%), TPR 99.98% 유지.

---

### 6-2. SNS False Positive 진단

**발생 케이스 3건:**

| 케이스 | 소재 | Stage 1 결과 | 원인 진단 |
|---|---|---|---|
| 인스타 릴스 캡처 | 실제 영상 프레임 캡처 | FAKE (S1) | beauty filter + SNS 인코더 cascade 압축 |
| 김원호 선수 프로필 (구글) | 스튜디오 사진 | FAKE (S1) | 고화질 스튜디오 리터칭, CDN 재압축 |
| 본인 증명사진 (백현종) | 스튜디오 촬영 | FAKE (S1) | skin retouching + 스튜디오 조명 평탄화 |

**공통 원인 메커니즘:**
1. **GAN 주파수 시그니처 모방**: 스튜디오 후처리 + 뷰티 필터 → 피부 질감 균일화 → GAN 생성 이미지와 유사한 주파수 특성
2. **다중 JPEG 압축 cascade**: SNS 업로드 코덱 → 스크린샷 → JPEG 저장 과정에서 아티팩트 누적
3. **학습-배포 도메인 gap**: 학습셋(FF++/DFDC)에 고화질 화보/SNS 캡처 없음

**기각된 가설:**
- PNG vs JPG 포맷 차이: 실측으로 반증. testreal2(PNG) v13=0.361, testreal1(JPG) v13=0.461 → PNG가 더 낮음.
- Stage 2 (v13)가 문제: 실측으로 반증. testreal2의 v13 Stage 2만 보면 prob 0.361로 통과 가능.

---

### 6-3. cuDNN 환경 트러블슈팅 (★ 재발 방지 필수 ★)

**증상:**
```
Could not load library libcudnn_cnn_train.so.8.
Error: undefined symbol: _ZN5cudnn14cublasSaxpy_v2EP13cublasContextiPKfS3_iPfi
RuntimeError: FIND was unable to find an engine to execute this computation
  (at loss.backward())
```

**원인:**
- `.venv/bin/python`은 `/usr/bin/python`의 심볼릭 링크
- bare 경로로 실행하면 `~/.local/lib/python3.9/site-packages`의 torch 2.2.2가 shadow
- torch 2.2.2 + cuDNN 8902(시스템) 조합은 심볼 충돌 → backward에서 폭발

**시도한 실패 경로:**
1. `cudnn.enabled = False` (CLAUDE.md §4 워크어라운드) → 여전히 crash. 심볼 자체가 없음
2. `/home/t26106/.venv/bin/python train_vN.py` 절대경로 직접 실행 → venv 격리 안 됨 (symlink 문제)

**올바른 해결책:**
```bash
nohup bash -c "source /home/t26106/.venv/bin/activate && python -u train_vN.py" > train_vN.log 2>&1 &
```
→ `source activate` 로만 venv가 제대로 격리됨.

**검증 방법:** 로그 첫 줄에 아래 확인:
```
torch=2.8.0+cu128 | cudnn=91002 | enabled=True
```

**환경 대조표:**
| | 경로 | torch | cuDNN | cudnn.enabled | 학습 가능 |
|---|---|---|---|---|---|
| System | `/usr/bin/python` | 2.2.2+cu121 | 8902 | False 필수 | ❌ (권장 X) |
| venv | `source activate 후 python` | 2.8.0+cu128 | 91002 | True 가능 | ✅ |

---

### 6-4. v14 설계 및 학습 (SNS Augmentation Pilot)

**설계 목적:** v13 → v14는 augmentation 변수 하나만 추가. 변수 통제 원칙 준수.

**추가된 augmentation 클래스:**

| 클래스 | 역할 | 주요 파라미터 |
|---|---|---|
| `RandomStrongGaussianBlur` | SNS 필터 시뮬 | σ ∈ [0.8, 2.5] |
| `RandomBilateralFilter` | beauty filter/skin smoothing 시뮬 | d∈{5,7,9}, σ_color/space∈[35,80] |
| `RandomMedianBlur` | 노이즈 제거 필터 시뮬 | k∈{3,5} |
| `OneOfSmoothing` | 위 3가지 중 하나 확률적 적용 | p=0.6 |
| `RandomDoubleJPEG` | 2회 JPEG 인코딩 (SNS cascade) | q1∈[40,90], q2∈[40,90], p=1.0 |

**학습 설정:** 모두 v13과 동일 (FreqOnlyXception, DF40 데이터, lr=1e-4, bs=16, focal loss γ=2.0)

**학습 결과 (중간, epoch 3 best):**
```
Epoch 1: val_auc=1.0000  ← 너무 이른 saturation (in-domain 너무 쉬움)
Epoch 2: val_auc=0.9999
Epoch 3: val_auc=1.0000
```

---

### 6-5. SNS OOD 정량 평가 결과 (eval_sns_ood.py)

**평가 환경:** GPU 0, venv, epoch 3 best checkpoint

**전체 결과:**
```
Image        |  v7    v8    v10  |  S1    |  v11   v12   v13   v14  | S2(13) S2(14) | Final13 Final14
testreal1    | 0.841 0.587 0.747 | 0.699  | 0.566 0.755 0.461 0.517 | 0.549  0.577  | FAKE(S1) FAKE(S1)
testreal2    | 0.970 0.656 0.233 | 0.554  | 0.188 0.163 0.361 0.459 | 0.260  0.301  | FAKE(S1) FAKE(S1)
testreal3    | 0.973 0.557 0.681 | 0.711  | 0.695 0.843 0.568 0.615 | 0.666  0.687  | FAKE(S1) FAKE(S1)
```

**v14 vs v13 단독 비교:**
| 이미지 | v13 | v14 | 변화 |
|---|---|---|---|
| testreal1 | 0.461 | 0.517 | **+0.056 악화** |
| testreal2 | 0.361 | 0.459 | **+0.098 악화** |
| testreal3 | 0.568 | 0.615 | **+0.047 악화** |

**핵심 발견:**
1. **3장 모두 Stage 1에서 FAKE 판정** → Stage 2까지 도달조차 못함. v14 효과가 cascade 결정에 아무 영향 없음.
2. **v7이 진짜 원흉:** testreal1=0.841, testreal2=0.970, testreal3=0.973. 모든 SNS 이미지를 극단적 확신으로 fake로 봄.
3. **v14가 오히려 악화:** SNS aug가 real 이미지를 더 fake-like로 학습시켰을 가능성. 또는 epoch 3 best로 학습 부족.
4. **v10(FreqOnly)은 상대적으로 선방:** testreal2에서 v10=0.233으로 가장 낮음. 주파수 전용 모델이 RGB 모델보다 SNS에 내성 있음.

**결론:** Stage 2 retrain보다 **Stage 1 (v7) retrain이 선행 과제**.

---

### 6-6. 생성된 도구/스크립트

| 파일 | 역할 |
|---|---|
| `train_v14.py` | v13 + SNS augmentation 학습 스크립트 |
| `eval_sns_ood.py` | testreal{1,2,3} 대상 v13 vs v14 정량 비교 |
| `GEMINI.md` | Gemini CLI용 학과서버/프로젝트 가이드 |

---

## 7. v15 학습 현황 (Gemini CLI 착수, 2026-05-11)

**목적:** v7(Xception RGB) + SNS-aware augmentation → v15 (Stage 1 retrain)

**학습 상태 (최신 로그):**
```
[v15] Xception RGB + SNS-aug | device=cuda:1
torch=2.8.0+cu128 | cudnn=91002 | enabled=True
train: 149,222 | val: 15,439

[Epoch 1/30] lr=5.00e-05 | TrainLoss: 0.0345 | ValAUC: 0.9498
[Epoch 2/30] lr=1.00e-04 | TrainLoss: 0.0210 | ValAUC: 0.9557
[Epoch 3/30] lr=1.00e-04 | TrainLoss: 0.0158 | ValAUC: 0.9684
[Epoch 4/30] lr=9.97e-05 | TrainLoss: 0.0132 | ValAUC: 0.9685
[Epoch 5/30] lr=9.88e-05 | TrainLoss: 0.0114 | ValAUC: 0.9723
[Epoch 6/30] lr=9.72e-05 | TrainLoss: 0.0100 | ValAUC: 0.9722
[Epoch 7/30] lr=9.51e-05 | TrainLoss: 0.0089 | ValAUC: 0.9692
[Epoch 8/30] lr=9.24e-05 | TrainLoss: 0.0078 | ValAUC: 0.9728 ⭐ best
```

**관찰:**
- v7 원본 val_auc는 0.9793이었음 → v15 현재 0.9728 (epoch 8). 아직 학습 중이므로 최종 판단 보류.
- v14(DF40, val_auc 1.0 saturation)와 달리 v15는 FF++/DFDC 학습이라 val_auc가 점진적으로 오름 → 정상적인 학습 곡선.
- 4-way Sampler 사용 (dfdc_0/1, ff_0/1 각 그룹별 균형 샘플링).

**평가 예정:** 학습 완료 후 eval_sns_ood.py로 testreal{1,2,3} 재평가. v7(0.841/0.970/0.973) → 유의미한 하락이면 Stage 1 FP 문제 해소.

---

## 8. 다음 단계 의사결정 트리

```
v15 평가 (testreal{1,2,3})
  ├─ v15 testreal prob 유의미하게 하락 (예: < 0.6)
  │    → Stage 1 S1_prob ≥ 0.5 통과 기대
  │    → v16(v8+SNS-aug), v17(v10+SNS-aug) 순차 착수
  │    → 최종 ensemble weight 재튜닝
  │
  └─ v15도 여전히 높음 (≥ 0.8)
       → augmentation 강도/종류 재검토
       → Stage 1 threshold 0.5 → 0.55~0.6 상향 검토
       → Stage 2가 강하게 REAL이면 S1 override하는 bypass logic 검토
```

---
*마지막 업데이트: 2026-05-11 (Claude 세션 + Gemini CLI 세션 통합)*
