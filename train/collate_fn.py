import torch

def collate_fn(batch):
    max_n_patches = max(item['flow_patches'].shape[0] for item in batch)
    patch_len = batch[0]['flow_patches'].shape[-1]
    B = len(batch)
    flow_patches_padded = torch.zeros((B, max_n_patches, 1, patch_len), dtype=torch.float32)
    mask_padded = torch.zeros((B, max_n_patches), dtype=torch.float32)

    for i, item in enumerate(batch):
        n_patches = item['flow_patches'].shape[0]
        flow_patches_padded[i, :n_patches] = item['flow_patches']
        mask_padded[i, :n_patches] = 1.0

    age = torch.tensor([item['age'] for item in batch], dtype=torch.float32)
    sex = torch.tensor([item['sex'] for item in batch], dtype=torch.float32)
    smoking = torch.tensor([item['smoking'] for item in batch], dtype=torch.float32)
    label = torch.tensor([item['label'] for item in batch], dtype=torch.float32)  # [B]
    patient_id = torch.tensor([item['patient_id'] for item in batch], dtype=torch.long)

    batch_dict = {
        'flow_patches': flow_patches_padded,
        'mask': mask_padded,
        'age': age.unsqueeze(1),       # [B,1]
        'sex': sex.unsqueeze(1),       # [B,1]
        'smoking': smoking.unsqueeze(1),  # [B,1]
        'label': label,                # [B]
        'patient_id': patient_id
    }
    return batch_dict
