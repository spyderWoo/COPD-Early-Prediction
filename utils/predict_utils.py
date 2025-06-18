################################################################################
# predict_utils.py
# 학습된 모델 로드 및 새로운 샘플(CSV) 예측 유틸리티 함수
################################################################################

import os
import torch
import numpy as np
from model.spiro_predictor import SpiroPredictor
from model.spiro_explainer import compute_concavity_features

def load_models(checkpoint_dir, cfg):
    """간단한 SpiroPredictor 모델 로드"""
    device = torch.device(cfg['train']['device'])

    model = SpiroPredictor(input_dim=3 + 4,
                           hidden_dim=cfg['model']['explainer_hidden_dim']).to(device)
    ckpt_path = os.path.join(checkpoint_dir, "detection_best.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    return model

def predict_copd(sample_flow_patches, sample_clinical_feats, model, cfg):
    """간단한 SpiroPredictor 기반 COPD 확률 예측"""
    device = torch.device(cfg['train']['device'])
    model.eval()

    with torch.no_grad():
        sample_flow_patches = sample_flow_patches.to(device)
        concav = compute_concavity_features(sample_flow_patches, cfg['model']['patch_length']).to(device)

        age = torch.tensor(sample_clinical_feats[:, 0:1], dtype=torch.float32, device=device)
        sex = torch.tensor(sample_clinical_feats[:, 1:2], dtype=torch.float32, device=device)
        smoking = torch.tensor(sample_clinical_feats[:, 2:3], dtype=torch.float32, device=device)

        logit = model(age, sex, smoking, concav)
        prob = torch.sigmoid(logit).cpu().numpy().item()

    # Future risk 예측 모델이 없으므로 동일 확률 반환
    return prob, prob
