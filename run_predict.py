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
    model = load_models(cfg['output']['checkpoint_dir'], cfg)

    df = pd.read_csv(args.input_csv)
    results = []
    for _, row in df.iterrows():
        flow_patches = np.load(row['flow_path'])  # npy file (n_patches, patch_length)
        flow_tensor = torch.from_numpy(flow_patches).unsqueeze(1).float().unsqueeze(0)
        clinical = row[['AGE', 'SEX', 'SMOKING']].values.astype(np.float32).reshape(1, 3)
        prob_copd, prob_future = predict_copd(flow_tensor, clinical, model, cfg)
        results.append({
            'SEQN': row['SEQN'], 'prob_copd': prob_copd, 'prob_future': prob_future
        })

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print("예측 완료:", args.output_csv)

if __name__=="__main__":
    main()
