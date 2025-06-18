import os
import yaml
import torch
import pandas as pd

from preprocessing.nhanes_loader import NHANESDataset
from preprocessing.graph_builder import build_ontology_graph
from train.train_detection import train_detection
from train.train_prediction import train_prediction

def main():
    # 1) Load config
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2) Prepare NHANES dataset
    print("===== NHANES dataset loading =====")
    dataset = NHANESDataset(
        demo_path=cfg['data']['demo_path'],
        smq_path=cfg['data']['smq_path'],
        rdq_path=cfg['data']['rdq_path'],
        mcq_path=cfg['data']['mcq_path'],
        ocq_path=cfg['data']['ocq_path'],
        cotnal_path=cfg['data']['cotnal_path'],
        cbc_path=cfg['data']['cbc_path'],
        spx_g_path=cfg['data']['spx_g_path'],
        spxraw_g_path=cfg['data']['spxraw_g_path'],
        smoothing_sigma=cfg['data']['smoothing_sigma'],
        patch_length=cfg['data']['patch_length'],
    )
    print(f"✅ Total samples: {len(dataset)}")

    # 3) Build ontology graph
    print("===== building ontology graph =====")
    nhanes_tables = {
        'DEMO': pd.read_sas(cfg['data']['demo_path']),
        'SMQ': pd.read_sas(cfg['data']['smq_path']),
        'RDQ': pd.read_sas(cfg['data']['rdq_path']),
        'MCQ': pd.read_sas(cfg['data']['mcq_path']),
        'OCQ': pd.read_sas(cfg['data']['ocq_path']),
        'CBC': pd.read_sas(cfg['data']['cbc_path']),
        'COTNAL': pd.read_sas(cfg['data']['cotnal_path']),
    }
    graph_data = build_ontology_graph(nhanes_tables, cfg['ontology']['map_path'])
    print("✅ Ontology graph ready\n")

    # 4) COPD Detection training
    print("===== COPD Detection training =====")
    detection_model = train_detection(dataset, cfg)
    print("===== COPD Detection done =====\n")

    # 5) Early‐Prediction / Fusion training
    print("===== Fusion Model training =====")
    fusion_model = train_prediction(dataset, graph_data, cfg)
    print("===== Fusion Model done =====")

if __name__ == "__main__":
    main()
