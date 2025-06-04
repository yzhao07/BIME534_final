import random
import os
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import os

class TiffPatchDataset(Dataset):
    def __init__(self, image_dir):
        random.seed(42)
        all_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')]
        self.paths = random.sample(all_paths, 100)  # Randomly select 100 images

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.paths[idx]
