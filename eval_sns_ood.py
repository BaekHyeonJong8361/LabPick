"""
eval_sns_ood.py
===============
SNS OOD 평가셋(testreal1/2/3)에서 v13 vs v14 정량 비교.

- 입력: /home/t26106/deepfake/test_inputs/testreal{1.jpg, 2.png, 3.png}
- 처리: MediaPipe face detect → 1.3× square crop → 299 resize
- 출력:
    - 각 모델 단독 prob (v7, v8, v10, v11, v12, v13, v14)
    - Stage 1 cascade prob/decision  (v7+v8+v10, thr=0.5)
    - Stage 2 baseline prob/decision (v11+v12+v13, thr=0.55)
    - Stage 2 with v14 prob/decision (v11+v12+v14, thr=0.55)
    - Final cascade decision (baseline vs v14)

inference_server_v3.py의 Stage 1/2 weight, threshold, transform을 그대로 재사용.

실행:
    source /home/t26106/.venv/bin/activate
    python eval_sns_ood.py
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
import mediapipe as mp

# 프로젝트 모듈
from train_v6 import get_eval_transforms
from train_v8 import F3NetLite
from train_freq_only_v10 import FreqOnlyXception
from train_v11 import F3NetLiteV11
from dataset import RandomJPEGCompression


# ============ 설정 (inference_server_v3.py와 동일) ============
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1
                     else ("cuda" if torch.cuda.is_available() else "cpu"))
FACE_SIZE = 299
CROP_SCALE = 1.3

MODEL_PATHS = {
    "v7":  "saved_models/xception_best_v7_combined.pth",
    "v8":  "saved_models/f3netlite_best_v8_combined.pth",
    "v10": "saved_models/freq_only_best_v10.pth",
    "v11": "saved_models/f3netlite_best_v11_df40.pth",
    "v12": "saved_models/xception_best_v12_df40.pth",
    "v13": "saved_models/freq_only_best_v13_df40.pth",
    "v14": "saved_models/freq_only_best_v14_df40_aug.pth",
}

STAGE1_THRESHOLD = 0.5
STAGE2_THRESHOLD = 0.55
STAGE1_WEIGHTS = {"v7": 0.3, "v8": 1.0, "v10": 1.0}
STAGE2_WEIGHTS = {"v11": 1.0, "v12": 0.5, "v13": 1.5}
# v14 평가용: v13 자리에 v14 (동일 가중치 1.5)
STAGE2_WEIGHTS_V14 = {"v11": 1.0, "v12": 0.5, "v14": 1.5}

TEST_IMAGES = [
    ("testreal1", "/home/t26106/deepfake/test_inputs/testreal1.jpg"),
    ("testreal2", "/home/t26106/deepfake/test_inputs/testreal2.png"),
    ("testreal3", "/home/t26106/deepfake/test_inputs/testreal3.png"),
]


# ============ 모델 로딩 ============
def build_model(name: str):
    if name == "v7":
        return timm.create_model("xception", pretrained=False, num_classes=1)
    if name == "v8":
        return F3NetLite(img_size=299, num_bands=3, pretrained=False)
    if name == "v10":
        return FreqOnlyXception(img_size=299, num_bands=3, pretrained=False, dropout=0.3)
    if name == "v11":
        return F3NetLiteV11(img_size=299, num_bands=3, pretrained=False, dropout=0.5)
    if name == "v12":
        return timm.create_model("xception", pretrained=False, num_classes=1)
    if name == "v13":
        return FreqOnlyXception(img_size=299, num_bands=3, pretrained=False, dropout=0.3)
    if name == "v14":
        return FreqOnlyXception(img_size=299, num_bands=3, pretrained=False, dropout=0.3)
    raise ValueError(name)


def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"  ⏩ {name} ckpt 없음 — skip ({path})")
            continue
        m = build_model(name).to(DEVICE)
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
        m.load_state_dict(state)
        m.eval()
        models[name] = m
        print(f"  ✅ {name} loaded")
    return models


# ============ 얼굴 검출 + 크롭 ============
def detect_and_crop(image_path: str):
    """단일 얼굴 crop PIL 반환. (inference_server_v3.detect_and_crop과 동일 로직)"""
    mp_fd = mp.solutions.face_detection
    fd = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        # PIL fallback (PNG with alpha 등)
        pil = Image.open(image_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    H, W = rgb.shape[:2]

    res = fd.process(rgb)
    if not res.detections:
        print(f"  ⚠ 얼굴 미검출: {image_path}")
        return None

    det = max(res.detections, key=lambda d: d.score[0] if d.score else 0.0)
    bbox = det.location_data.relative_bounding_box
    x1 = bbox.xmin * W
    y1 = bbox.ymin * H
    x2 = (bbox.xmin + bbox.width) * W
    y2 = (bbox.ymin + bbox.height) * H

    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(w, h) * CROP_SCALE
    x1n = int(max(0, cx - side / 2))
    y1n = int(max(0, cy - side / 2))
    x2n = int(min(pil.width, cx + side / 2))
    y2n = int(min(pil.height, cy + side / 2))

    face = pil.crop((x1n, y1n, x2n, y2n)).resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)
    return face


# ============ 추론 ============
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


@torch.no_grad()
def run_model(model: nn.Module, tensor: torch.Tensor) -> float:
    """단일 샘플 prob(fake) 반환."""
    out = model(tensor.unsqueeze(0).to(DEVICE))
    if isinstance(out, (tuple, list)):
        out = out[0]
    logit = out.squeeze().detach().cpu().numpy().item()
    return float(sigmoid(logit)), float(logit)


def weighted_logit_avg(logits: dict, weights: dict) -> float:
    """가중 로짓 평균 → prob."""
    num = sum(weights[k] * logits[k] for k in weights if k in logits)
    den = sum(weights[k] for k in weights if k in logits)
    if den == 0:
        return 0.0
    return float(sigmoid(num / den))


# ============ 메인 ============
def main():
    print(f"[init] device={DEVICE}, torch={torch.__version__}")
    print("[모델 로딩]")
    models = load_models()

    # transform
    transform_stage1 = get_eval_transforms(img_size=FACE_SIZE)
    transform_stage2 = transforms.Compose([
        transforms.Resize((FACE_SIZE, FACE_SIZE)),
        RandomJPEGCompression(min_quality=80, max_quality=80),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("\n" + "=" * 100)
    print(f"{'Image':<12} | {'v7':>6} {'v8':>6} {'v10':>6} | {'S1':>7} | "
          f"{'v11':>6} {'v12':>6} {'v13':>6} {'v14':>6} | {'S2(13)':>7} {'S2(14)':>7} | "
          f"{'Final13':>8} {'Final14':>8}")
    print("=" * 100)

    rows = []
    for name, path in TEST_IMAGES:
        if not os.path.exists(path):
            print(f"{name}: 파일 없음 ({path})")
            continue
        face = detect_and_crop(path)
        if face is None:
            print(f"{name}: 얼굴 미검출 — skip")
            continue

        t1 = transform_stage1(face)
        t2 = transform_stage2(face)

        probs, logits = {}, {}
        for mname in ("v7", "v8", "v10"):
            if mname in models:
                probs[mname], logits[mname] = run_model(models[mname], t1)
        for mname in ("v11", "v12", "v13", "v14"):
            if mname in models:
                probs[mname], logits[mname] = run_model(models[mname], t2)

        # Stage 1
        s1_prob = weighted_logit_avg(logits, STAGE1_WEIGHTS)
        s1_fake = s1_prob >= STAGE1_THRESHOLD

        # Stage 2 baseline (v13)
        s2b_prob = weighted_logit_avg(logits, STAGE2_WEIGHTS)
        s2b_fake = s2b_prob >= STAGE2_THRESHOLD

        # Stage 2 with v14
        s2v_prob = weighted_logit_avg(logits, STAGE2_WEIGHTS_V14)
        s2v_fake = s2v_prob >= STAGE2_THRESHOLD

        # Final cascade decision:
        #   Stage1 FAKE → FAKE (classic), else Stage2 결정으로
        if s1_fake:
            final13 = "FAKE(S1)"
            final14 = "FAKE(S1)"
        else:
            final13 = "FAKE(S2)" if s2b_fake else "REAL"
            final14 = "FAKE(S2)" if s2v_fake else "REAL"

        def fmt(p): return f"{p:.3f}" if p is not None else "  -  "

        print(f"{name:<12} | "
              f"{fmt(probs.get('v7')):>6} {fmt(probs.get('v8')):>6} {fmt(probs.get('v10')):>6} | "
              f"{fmt(s1_prob):>7} | "
              f"{fmt(probs.get('v11')):>6} {fmt(probs.get('v12')):>6} "
              f"{fmt(probs.get('v13')):>6} {fmt(probs.get('v14')):>6} | "
              f"{fmt(s2b_prob):>7} {fmt(s2v_prob):>7} | "
              f"{final13:>8} {final14:>8}")

        rows.append({
            "image": name,
            "probs": probs,
            "stage1_prob": s1_prob,
            "stage2_baseline_prob": s2b_prob,
            "stage2_v14_prob": s2v_prob,
            "final_baseline": final13,
            "final_v14": final14,
        })

    print("=" * 100)
    print("\n[해석 가이드]")
    print(f"  - Stage 1 threshold = {STAGE1_THRESHOLD} (v7+v8+v10 weighted logit avg)")
    print(f"  - Stage 2 threshold = {STAGE2_THRESHOLD} (v11+v12+(v13|v14) weighted logit avg)")
    print(f"  - 진짜 인물 사진이므로 final이 REAL이면 통과, FAKE면 실패")
    print(f"  - v14 column이 v13 column 대비 prob 감소했으면 augmentation 효과 있음")

    # 요약
    print("\n[요약]")
    base_real = sum(1 for r in rows if r["final_baseline"] == "REAL")
    v14_real = sum(1 for r in rows if r["final_v14"] == "REAL")
    print(f"  Baseline (v13): {base_real}/{len(rows)} REAL")
    print(f"  With v14:       {v14_real}/{len(rows)} REAL")


if __name__ == "__main__":
    main()
