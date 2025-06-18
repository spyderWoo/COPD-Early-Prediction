import argparse
import yaml
import numpy as np
import pandas as pd
import torch

from utils.predict_utils import load_models, predict_copd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",  type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml","r"))
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    encoder, explainer, cb_det, cb_pred = load_models(cfg, device)

    df = pd.read_csv(args.input_csv)
    results = []
    for _, row in df.iterrows():
        prob_copd, prob_future = predict_copd(row, encoder, explainer, cb_det, cb_pred, cfg, device)
        results.append({
            'SEQN': row['SEQN'], 'prob_copd': prob_copd, 'prob_future': prob_future
        })

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print("예측 완료:", args.output_csv)

if __name__=="__main__":
    main()
