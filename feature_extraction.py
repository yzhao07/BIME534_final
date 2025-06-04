from conch.open_clip_custom import create_model_from_pretrained
from dataset import TiffPatchDataset
import os
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader

model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "./pytorch_model.bin")
device = "cuda:8"
model = model.to(device)
all_embeddings = []
all_label = []
for p in os.listdir("/scratch/individuals/yujie/class_final/data/CRC-VAL-HE-7K"):
    dataset = TiffPatchDataset(os.path.join("/scratch/individuals/yujie/class_final/data/CRC-VAL-HE-7K",p))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.inference_mode():
        for imgs, paths in tqdm.tqdm(loader):
            #print(imgs.shape)
            imgs = imgs.to(device)
            image_embs = model.encode_image(imgs, proj_contrast=False, normalize=False)
            #print(image_embs.shape)
            all_embeddings.append(image_embs.cpu())
            all_label.extend([p] * imgs.shape[0])

# Save or concatenate
features = torch.cat(all_embeddings, dim=0).numpy() # shape: (N_patches, 768)
print(features.shape)
all_label = np.array(all_label)
print(all_label.shape)
np.save("features.npy", features)
np.save("labels.npy", all_label)