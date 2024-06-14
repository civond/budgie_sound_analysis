import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torchvision.models as models
from torchvision.transforms import v2
import pandas as pd

from utils import *

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 2
NUM_WORKERS = 8
PIN_MEMORY = True
LOAD_MODEL = False

dataset_pth = "meta.csv"

df = pd.read_csv(dataset_pth)
#df['label'].replace({1: 0, 2: 1}, inplace=True) # Reformat
df.replace({'label': {1: 0, 2: 1}}, inplace=True)

def main():
    # Test
    test_df = df[df['fold'] == 5]
    test_df.reset_index(drop=True, inplace=True)

    # Test
    train = False
    test_transform = v2.Compose([
                    v2.ToTensor(),
                    v2.Normalize(
                            mean=[0.4914, 0.4822, 0.4465],
                            std= np.sqrt([1.0, 1.0, 1.0]) # variance is std**2
                        )
                ])

    test_loader = get_loader(
            test_df,
            BATCH_SIZE,
            test_transform,
            NUM_WORKERS,
            train,
            PIN_MEMORY
        )
    num_classes = 2
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(1280, num_classes)
    model = model.to(DEVICE)


    print("Generating predictions")
    preds = predict(test_loader, 
                    model, 
                    device=DEVICE)
    
    test_df['preds'] = preds
    temp_preds_path = f"csv/test_preds.csv"
    print(f"Writing {temp_preds_path}")
    test_df.to_csv(temp_preds_path, sep=',', index=False)

if __name__ == "__main__":
    main()