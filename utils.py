from dataset import *
from torch.utils.data import DataLoader, ConcatDataset
import torch

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("-> Saving checkpoint.")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("-> Loading checkpoint.")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loader(voc_dir,
               noise_dir,
               batch_size,
               transform,
               num_workers=12,
               train=True,
               pin_memory=True):
    
    voc_ds = ImageDataset(
        voc_dir, 
        train=train,
        transform=transform
        )
    noise_ds = ImageDataset(
        noise_dir, 
        train=train,
        transform=transform
        )
    train_dataset = ConcatDataset([voc_ds, noise_ds])

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    return loader

def check_accuracy(loader, model, device="cuda"):
    correct = 0
    total = 0

    model.eval()
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total
    print(f"Accuracy: {accuracy}")
    return accuracy