import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import os

from evaluation.metrics import compute_metrics
from evaluation.visualize import plot_roc, plot_pr
from model.fusion_model import FusionModel
from model.ontology_gnn import OntologyGNNEncoder
from train.collate_fn import collate_fn

def train_prediction(dataset, graph_data, cfg):
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    batch_size = int(cfg['data']['batch_size'])
    epochs     = int(cfg['train']['epochs_prediction'])
    lr         = float(cfg['train']['learning_rate'])
    test_size  = float(cfg['train']['test_size'])
    seed       = int(cfg['train']['random_seed'])

    total_len = len(dataset)
    val_len   = int(total_len * test_size)
    train_len = total_len - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    gnn_encoder = OntologyGNNEncoder(
        in_channels=graph_data['patient'].x.shape[1],
        hidden_channels=32,
        out_channels=32
    ).to(device)
    fusion_model = FusionModel(
        gnn_dim=32,
        spiro_dim=cfg['model']['explainer_hidden_dim'],
        hidden_dim=64
    ).to(device)

    optimizer = torch.optim.Adam(list(gnn_encoder.parameters()) + list(fusion_model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    os.makedirs(cfg['output']['checkpoint_dir'], exist_ok=True)

    # patient2id 매핑 로드 (SEQN → 노드 인덱스)
    patient2id = graph_data.meta_info['patient2id']

    for epoch in range(1, epochs + 1):
        gnn_encoder.train()
        fusion_model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            flow    = batch['flow_patches'].to(device)
            age     = batch['age'].to(device)
            sex     = batch['sex'].to(device)
            smoking = batch['smoking'].to(device)
            label   = batch['label'].to(device).float()   # <--- FIXED HERE

            # patient_id(SEQN) -> 그래프 노드 인덱스 변환
            batch_patient_seqns = batch['patient_id']
            if isinstance(batch_patient_seqns, torch.Tensor):
                batch_patient_seqns = batch_patient_seqns.cpu().numpy()
            batch_graph_index = [patient2id[int(seqn)] for seqn in batch_patient_seqns]
            batch_graph_index = torch.tensor(batch_graph_index, dtype=torch.long, device=device)

            optimizer.zero_grad()
            gnn_emb = gnn_encoder(graph_data)['patient']
            patient_emb = gnn_emb[batch_graph_index]
            # 임시: 스파이로그램 임베딩 placeholder
            spiro_emb = torch.zeros(patient_emb.shape[0], cfg['model']['explainer_hidden_dim'], device=device)

            output = fusion_model(patient_emb, spiro_emb, age, sex, smoking)
            loss = criterion(output.squeeze(), label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        gnn_encoder.eval()
        fusion_model.eval()
        y_true_list, y_prob_list = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Valid]"):
                flow    = batch['flow_patches'].to(device)
                age     = batch['age'].to(device)
                sex     = batch['sex'].to(device)
                smoking = batch['smoking'].to(device)
                label   = batch['label'].to(device).float()   # <--- FIXED HERE

                batch_patient_seqns = batch['patient_id']
                if isinstance(batch_patient_seqns, torch.Tensor):
                    batch_patient_seqns = batch_patient_seqns.cpu().numpy()
                batch_graph_index = [patient2id[int(seqn)] for seqn in batch_patient_seqns]
                batch_graph_index = torch.tensor(batch_graph_index, dtype=torch.long, device=device)

                gnn_emb = gnn_encoder(graph_data)['patient']
                patient_emb = gnn_emb[batch_graph_index]
                spiro_emb = torch.zeros(patient_emb.shape[0], cfg['model']['explainer_hidden_dim'], device=device)

                output = fusion_model(patient_emb, spiro_emb, age, sex, smoking)
                prob = torch.sigmoid(output.squeeze())

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
            ckpt_path = os.path.join(cfg['output']['checkpoint_dir'], "fusion_best.pth")
            torch.save({
                'gnn_encoder': gnn_encoder.state_dict(),
                'fusion_model': fusion_model.state_dict(),
            }, ckpt_path)

    # 결과 그래프 디렉토리 자동 생성
    os.makedirs(cfg['output']['figures_dir'], exist_ok=True)
    plot_roc(y_true_all, y_prob_all, os.path.join(cfg['output']['figures_dir'], "roc_prediction.png"))
    plot_pr(y_true_all, y_prob_all, os.path.join(cfg['output']['figures_dir'], "pr_prediction.png"))

    print("✅ Fusion 모델 학습 완료")
    return fusion_model
