import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import pandas as pd

# .csv paths
pred_paths = ["csv/0_color_preds2.csv",
              "csv/1_color_preds2.csv",
              "csv/2_color_preds2.csv",
              "csv/3_color_preds2.csv",
              "csv/4_color_preds2.csv"]

valid_paths = ["csv/0_color_valid2.csv",
              "csv/1_color_valid2.csv",
              "csv/2_color_valid2.csv",
              "csv/3_color_valid2.csv",
              "csv/4_color_valid2.csv"]

# pred stats
i = 0
plt.figure(2, figsize=(6,7))
for path in pred_paths:
    df = pd.read_csv(path)
    labels = df['label']
    preds = df['preds']

    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    plt.subplot(3,2,i+1)
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                cbar=False,
                xticklabels=['Voc.', 'Noise'], 
                yticklabels=['Voc.', 'Noise'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f"acc. = {np.round(acc,4)}")
    i+=1
plt.tight_layout()
plt.savefig("csv/cm_color.png")

# epoch stats
plt.figure(1, figsize=(6,5))
for path in valid_paths:
    df = pd.read_csv(path)
    train_loss = df['train_loss']
    train_acc = df['train_acc']
    val_loss = df['val_loss']
    val_acc = df['val_acc']

    plt.subplot(2,2,1)
    plt.plot(train_loss, linewidth=1)
    plt.subplot(2,2,2)
    plt.plot(train_acc, linewidth=1)
    plt.subplot(2,2,3)
    plt.plot(val_loss, linewidth=1)
    plt.subplot(2,2,4)
    plt.plot(val_acc, linewidth=1)

plt.subplot(2,2,1)
plt.title("Train Loss")
plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.grid(True)
plt.xlim(0,14)
plt.ylim(0,0.65)

plt.subplot(2,2,2)
plt.title("Train Acc.")
plt.xlabel('Epoch')
plt.ylabel("Accuracy")
plt.grid(True)
plt.xlim(0,14)
plt.ylim(0.65,1)

plt.subplot(2,2,3)
plt.title("Val. Loss")
plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.grid(True)
plt.xlim(0,14)
plt.ylim(0,0.65)

plt.subplot(2,2,4)
plt.title("Val. Acc.")
plt.xlabel('Epoch')
plt.ylabel("Accuracy")
plt.grid(True)
plt.xlim(0,14)
plt.ylim(0.65,1)

plt.tight_layout()
plt.savefig("csv/stats.png")