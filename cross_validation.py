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
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 15
NUM_WORKERS = 12
PIN_MEMORY = True
LOAD_MODEL = False

dataset_pth = "meta_color3.csv"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_acc = 0
    total_loss = 0
    num_batches = len(loader)
    
    for batch_idx, (data, labels) in enumerate(loop):
        data = data.to(device=DEVICE)
        labels = labels.to(device=DEVICE)
        #print(f"Len Labels: {len(labels)}")

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, labels)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Get loss
        total_loss += loss.item()
        _, predicted = torch.max(predictions, 1)
        #print(f"Preds: {predicted}")
        #print(f"labels: {labels}")
        correct = (predicted == labels).sum().item()
        #print(correct)
        """total += labels.size(0)
        correct += (predicted == labels).sum().item()"""

        #print((predicted == labels))
    
        #print(total)
        
        accuracy = correct / len(labels)
        #print(accuracy)
        total_acc += accuracy
        

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
    avg_acc = total_acc / num_batches
    avg_loss = total_loss / num_batches
    print(f"Train Avg_Acc: {avg_acc}, Avg_Loss: {avg_loss}")
    return avg_acc, avg_loss

def main():
    # Load the model
    cross_valid_sets = [
        [4, [0,1,2,3]],
        [3, [0,1,2,4]],
        [2, [0,1,4,3]],
        [1, [0,4,2,3]],
        [0, [4,1,2,3]]
    ]
    
    for set in cross_valid_sets:
        train_running_loss = []
        train_running_acc = []
        val_running_loss = []
        val_running_acc = []
        
        print(f"Running: {set[1]}, Valid: {set[0]}")
        num_classes = 2
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(1280, num_classes)
        model = model.to(DEVICE)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scaler = torch.cuda.amp.GradScaler()

        # Transforms
        train_transform = v2.Compose([
            v2.ToTensor(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std= np.sqrt([1.0, 1.0, 1.0]) # variance is std**2
            )
            ])
        
        test_transform = v2.Compose([
                v2.ToTensor(),
                v2.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std= np.sqrt([1.0, 1.0, 1.0]) # variance is std**2
                    )
            ])
        
        # Import dataset
        df = pd.read_csv(dataset_pth)
        df.replace({'label': {1: 0, 2: 1}}, inplace=True) # Reformat labels

        # Train
        train_df = df[df['fold'].isin(set[1])]
        train_df.reset_index(drop=True, inplace=True)

        # Valid
        valid_df = df[df['fold'] == int(set[0])]
        valid_df.reset_index(drop=True, inplace=True)
        
        # Test
        test_df = df[df['fold'] == 5]
        test_df.reset_index(drop=True, inplace=True)

        # Training
        train = True
        shuffle=False

        train_loader = get_loader(
                train_df,
                BATCH_SIZE,
                train_transform,
                NUM_WORKERS,
                train,
                shuffle,
                PIN_MEMORY
            )
        
        # Validation
        train = False
        valid_loader = get_loader(
                valid_df,
                BATCH_SIZE,
                test_transform,
                NUM_WORKERS,
                train,
                shuffle,
                PIN_MEMORY
            )
        
        # Test
        train = False
        shuffle=False
        test_loader = get_loader(
                test_df,
                BATCH_SIZE,
                test_transform,
                NUM_WORKERS,
                train,
                shuffle,
                PIN_MEMORY
            )
        
        if LOAD_MODEL: 
            print("Loading model.")
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch: {epoch}")
            [train_acc, train_loss] = train_fn(train_loader, model, optimizer, loss_fn, scaler)
            # Save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            #save_checkpoint(checkpoint)
            [val_acc, val_loss] = validate(valid_loader, 
                            model,
                            loss_fn,
                            device=DEVICE)
            
            train_running_loss.append(train_loss)
            train_running_acc.append(train_acc)
            val_running_loss.append(val_loss)
            val_running_acc.append(val_acc)
        
        # Predict
        print("Generating predictions")
        preds, labels = predict(test_loader, 
                        model, 
                        device=DEVICE)
        
        temp_data = {
            "train_loss" : train_running_loss,
            "train_acc" : train_running_acc, 
            "val_loss" : val_running_loss,
            "val_acc" : val_running_acc
            }
        
        #print(temp_data)
        temp_df = pd.DataFrame(temp_data)
        temp_csv_path = f"csv/{set[0]}_color_valid2.csv"
        temp_preds_path = f"csv/{set[0]}_color_preds2.csv"
        
        print(f"\tWriting {temp_csv_path}")
        temp_df.to_csv(temp_csv_path, sep=',', index=False)
        
        print(f"\tWriting {temp_preds_path}")
        test_df['preds'] = preds
        test_df['label2'] = labels
        
        columns_to_keep = ['onset', 'offset','label', 'preds']
        test_df = test_df.drop(df.columns.difference(columns_to_keep), axis=1)
        test_df.to_csv(temp_preds_path, sep=',', index=False)

if __name__ == "__main__":
    main()