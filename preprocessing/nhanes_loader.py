import os
import numpy as np
import pandas as pd
import pyreadstat
import torch
from torch.utils.data import Dataset
from preprocessing.smoother import gaussian_smooth
from preprocessing.flow_converter import time_to_flow, construct_flow_volume

class NHANESDataset(Dataset):
    def __init__(self,
                 demo_path, smq_path, rdq_path, mcq_path,
                 ocq_path, cotnal_path, cbc_path,
                 spx_g_path, spxraw_g_path,
                 smoothing_sigma=2.0, patch_length=128):
        super().__init__()
        self.patch_length = patch_length
        self.smoothing_sigma = smoothing_sigma

        # 1) DEMO
        df_demo, _ = pyreadstat.read_xport(demo_path)
        df_demo = df_demo[['SEQN','RIDAGEYR','RIAGENDR']].rename(
            columns={'RIDAGEYR':'AGE','RIAGENDR':'GENDER'})
        df_demo['SEX'] = (df_demo['GENDER']==1).astype(int)
        self.demo_df = df_demo

        # 2) SMQ
        df_smq, _ = pyreadstat.read_xport(smq_path)
        df_smq = df_smq[['SEQN','SMQ020']]
        df_smq['SMOKING'] = (df_smq['SMQ020']==1).astype(int)
        self.smq_df = df_smq

        # 3) RDQ, MCQ, OCQ, COTNAL, CBC
        self.rdq_df, _ = pyreadstat.read_xport(rdq_path)
        self.mcq_df, _ = pyreadstat.read_xport(mcq_path)
        self.ocq_df, _ = pyreadstat.read_xport(ocq_path)
        self.cotnal_df, _ = pyreadstat.read_xport(cotnal_path)
        self.cbc_df, _ = pyreadstat.read_xport(cbc_path)

        # 4) SPX_G
        df_spx, _ = pyreadstat.read_xport(spx_g_path)
        df_spx = df_spx[['SEQN','SPXNFEV1','SPXNFVC']].rename(
            columns={'SPXNFEV1':'FEV1_ml','SPXNFVC':'FVC_ml'})
        df_spx['FEV1'] = df_spx['FEV1_ml']/1000.0
        df_spx['FVC']  = df_spx['FVC_ml']/1000.0
        df_spx['FEV1_FVC'] = df_spx['FEV1']/df_spx['FVC']
        df_spx = df_spx[['SEQN','FEV1_FVC']]
        self.spx_g_df = df_spx

        # 5) 병합 및 라벨 생성
        df = df_demo
        for merge_df in [df_smq, self.rdq_df, self.mcq_df, self.ocq_df, self.cotnal_df, self.cbc_df, df_spx]:
            df = pd.merge(df, merge_df, on='SEQN', how='inner')
        df['LABEL'] = (df['FEV1_FVC'] < 0.7).astype(int)
        self.merged_df = df
        self.summary_df = df.reset_index(drop=True)

        # 6) SPXRAW (곡선 신호 로드 및 파싱)
        self.curves = {}
        if os.path.exists(spxraw_g_path):
            df_raw, _ = pyreadstat.read_sas7bdat(spxraw_g_path)
            if 'SPXRAW' in df_raw.columns:
                for _, row in df_raw.iterrows():
                    seqn = row['SEQN']
                    raw_val = row['SPXRAW']
                    if isinstance(raw_val, (bytes, bytearray)):
                        raw_str = raw_val.decode('utf-8')
                    else:
                        raw_str = str(raw_val)
                    if any(c.isalpha() for c in raw_str.strip()):
                        continue
                    try:
                        vals = np.fromstring(raw_str.replace(' ', ''), sep=',', dtype=np.float32)
                        if vals.size > 10:
                            self.curves[seqn] = vals
                    except Exception:
                        continue
        # 🔥 [핵심] SPXRAW가 존재하는 SEQN만 남기기!
        valid_seqns = set(self.curves.keys())
        self.summary_df = self.summary_df[self.summary_df['SEQN'].isin(valid_seqns)].reset_index(drop=True)

    def __len__(self):
        return len(self.summary_df)

    def __getitem__(self, idx):
        r = self.summary_df.iloc[idx]
        seqn = r['SEQN']
        age    = torch.tensor([r['AGE']],dtype=torch.float32)
        sex    = torch.tensor([r['SEX']],dtype=torch.float32)
        smoking= torch.tensor([r['SMOKING']],dtype=torch.float32)
        label  = torch.tensor([r['LABEL']],dtype=torch.float32)

        curve = self.curves[seqn]  # curve는 반드시 존재
        cs    = gaussian_smooth(curve, sigma=self.smoothing_sigma)
        flow  = time_to_flow(cs, dt=0.01)
        vol   = cs[:-1]
        fv    = construct_flow_volume(vol, flow)
        n     = fv.shape[0]
        n_p   = int(np.ceil(n/self.patch_length))
        pad   = np.zeros((n_p*self.patch_length,2),dtype=np.float32)
        pad[:n] = fv
        pats  = pad.reshape(n_p,self.patch_length,2)
        fp    = torch.from_numpy(pats[:,:,1]).unsqueeze(1)
        mask  = torch.ones(n_p,dtype=torch.float32)

        return {
            'flow_patches': fp,
            'mask':         mask,
            'age':          age,
            'sex':          sex,
            'smoking':      smoking,
            'label':        label,
            'patient_id':   int(seqn)
        }
