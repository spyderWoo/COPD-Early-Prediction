# preprocessing/graph_builder.py
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import yaml

def build_ontology_graph(nhanes_df_dict, ontology_map_path):
    """
    NHANES 데이터 프레임들(NHANES 변수들)과 변수-개념 매핑파일을 기반으로
    HeteroData 이형 그래프를 구성한다.
    
    Args:
        nhanes_df_dict: dict of {table_name: pd.DataFrame}
        ontology_map_path: str, YAML 파일 경로 (변수 → 개념 맵핑)
    
    Returns:
        torch_geometric.data.HeteroData
    """
    data = HeteroData()
    
    # 1) 변수-개념 매핑 로드
    with open(ontology_map_path, 'r', encoding='utf-8') as f:
        variable_to_concept = yaml.safe_load(f)

    # 2) 노드 목록 수집
    patient_nodes = set()
    variable_nodes = set(variable_to_concept.keys())
    concept_nodes = set(variable_to_concept.values())

    # 노드 ID 매핑
    concept2id = {c: i for i, c in enumerate(sorted(concept_nodes))}
    variable2id = {v: i for i, v in enumerate(sorted(variable_nodes))}
    patient2id = {}  # SEQN → 정수 ID (0~)

    # 3) 환자-변수 간선 구성
    edge_index_p2v = [[], []]
    edge_weight_p2v = []

    next_pid = 0
    for table_name, df in nhanes_df_dict.items():
        for _, row in df.iterrows():
            seqn = int(row['SEQN'])
            if seqn not in patient2id:
                patient2id[seqn] = next_pid
                next_pid += 1
            pid = patient2id[seqn]

            for var in variable_nodes:
                if var in row and pd.notnull(row[var]):
                    val = row[var]
                    if isinstance(val, (int, float)) and val != 0:
                        edge_index_p2v[0].append(pid)
                        edge_index_p2v[1].append(variable2id[var])
                        edge_weight_p2v.append(float(val))
    
    # Tensor 변환
    data['patient'].x = torch.zeros(len(patient2id), 1)
    data['variable'].x = torch.nn.Parameter(torch.randn(len(variable2id), 32))
    data['concept'].x = torch.nn.Parameter(torch.randn(len(concept2id), 32))

    data['variable', 'has_value', 'patient'].edge_index = torch.tensor(edge_index_p2v[::-1], dtype=torch.long)
    data['variable', 'has_value', 'patient'].edge_attr = torch.tensor(edge_weight_p2v, dtype=torch.float)

    data['variable', 'has_value_rev', 'patient'].edge_index = torch.tensor(edge_index_p2v[::-1], dtype=torch.long)
    data['variable', 'has_value_rev', 'patient'].edge_attr = torch.tensor(edge_weight_p2v, dtype=torch.float)

    # 4) 변수-개념 연결
    edge_index_c2v = [[], []]
    for var, concept in variable_to_concept.items():
        vid = variable2id[var]
        cid = concept2id[concept]
        edge_index_c2v[0].append(cid)
        edge_index_c2v[1].append(vid)
    
    data['concept', 'belongs_to', 'variable'].edge_index = torch.tensor(edge_index_c2v, dtype=torch.long)
    data['variable', 'belongs_to_rev', 'concept'].edge_index = torch.tensor(edge_index_c2v[::-1], dtype=torch.long)

    # ID 매핑 저장 (필요 시 외부로 전달)
    data.meta_info = {
        'patient2id': patient2id,
        'variable2id': variable2id,
        'concept2id': concept2id
    }
    print("All node types:", list(data.node_types))
    print("All edge types:", list(data.edge_types))
    print("'patient' node shape:", data['patient'].x.shape)
    print("'variable' node shape:", data['variable'].x.shape)
    print("'concept' node shape:", data['concept'].x.shape)
    print("'variable', 'has_value', 'patient' edges:", data['variable', 'has_value', 'patient'].edge_index.shape)

    return data
