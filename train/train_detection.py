import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import os

from evaluation.metrics import compute_metrics
from evaluation.visualize import plot_roc, plot_pr
from model.spiro_predictor import SpiroPredictor
from model.spiro_explainer import compute_concavity_features
from train.collate_fn import collate_fn  # 패딩 collate_fn import

def train_detection(dataset, cfg):
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    batch_size = int(cfg['data']['batch_size'])
    epochs     = int(cfg['train']['epochs_detection'])
    lr         = float(cfg['train']['learning_rate'])
    test_size  = float(cfg['train']['test_size'])
    seed       = int(cfg['train']['random_seed'])

    total_len = len(dataset)
    val_len   = int(total_len * test_size)
    train_len = total_len - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = SpiroPredictor(input_dim=3 + 4, hidden_dim=cfg['model']['explainer_hidden_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    os.makedirs(cfg['output']['checkpoint_dir'], exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            age     = batch['age'].to(device)
            sex     = batch['sex'].to(device)
            smoking = batch['smoking'].to(device)
            label   = batch['label'].to(device).float()  # float32
            flow    = batch['flow_patches'].to(device)

            concav = compute_concavity_features(flow, cfg['model']['patch_length']).to(device)

            optimizer.zero_grad()
            logit = model(age, sex, smoking, concav)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        y_true_list, y_prob_list = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Valid]"):
                age     = batch['age'].to(device)
                sex     = batch['sex'].to(device)
                smoking = batch['smoking'].to(device)
                label   = batch['label'].to(device).float()
                flow    = batch['flow_patches'].to(device)

                concav = compute_concavity_features(flow, cfg['model']['patch_length']).to(device)
                logit = model(age, sex, smoking, concav)
                prob = torch.sigmoid(logit)

                y_true_list.append(label.cpu().numpy())
                y_prob_list.append(prob.cpu().numpy())

        y_true_all = np.concatenate(y_true_list)
        y_prob_all = np.concatenate(y_prob_list)
        y_pred_all = (y_prob_all >= 0.5).astype(int)

        metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)
        avg_loss = np.mean(train_losses)

        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | AUROC: {metrics['auroc']:.4f} | "
              f"AUPRC: {metrics['auprc']:.4f} | F1: {metrics['f1']:.4f}")

        if metrics['auroc'] > best_auc:
            best_auc = metrics['auroc']
            ckpt_path = os.path.join(cfg['output']['checkpoint_dir'], "detection_best.pth")
            torch.save(model.state_dict(), ckpt_path)

    # ---- 출력 디렉토리 자동 생성 추가 ----
    os.makedirs(cfg['output']['figures_dir'], exist_ok=True)

    plot_roc(y_true_all, y_prob_all, os.path.join(cfg['output']['figures_dir'], "roc_detection.png"))
    plot_pr(y_true_all, y_prob_all, os.path.join(cfg['output']['figures_dir'], "pr_detection.png"))

    print("✅ COPD 탐지 학습 완료")
    return model
