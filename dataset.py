from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision.transforms import v2
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_dir, train=True, transform = None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("L"))

        if self.transform is not None:
            image = self.transform(image)

        label = None
        if img_path.split('/')[1].split("_")[1] == "voc":
            print(img_path.split('/')[1].split("_")[1] == "voc")
            label = 0
        elif img_path.split('/')[1].split("_")[1] == "noise":
            print(img_path.split('/')[1].split("_")[1] == "voc")
            label = 1

        print(label)
        return image, label