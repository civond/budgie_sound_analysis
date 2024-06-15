from dataset import *
from torch.utils.data import DataLoader, ConcatDataset
import torch

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("-> Saving checkpoint.")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("-> Loading checkpoint.")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loader(dataframe,
               batch_size,
               transform,
               num_workers=12,
               train=True,
               shuffle=False,
               pin_memory=True):
    
    ds = ImageDataset(
        dataframe, 
        train=train,
        transform=transform
        )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader

def validate(loader, model, loss_fn, device="cuda"):
    total_loss = 0
    total_acc = 0
    model.eval()
    
    with torch.no_grad():
        batch_length = len(loader)


        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            total_loss += loss_fn(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            correct = (predicted == labels).sum().item()
            accuracy = (correct / len(labels))
            #print(accuracy)
            total_acc += accuracy
        
    avg_acc = total_acc / batch_length
    avg_loss = total_loss / batch_length # Compute the average loss across batches
    
    print(f"Validation Avg_Acc: {avg_acc}, Avg_Loss: {avg_loss}")
    return avg_acc, avg_loss.item()

def predict(loader, model, device="cuda"):
    preds_arr = []
    labels_arr = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            #labels = labels.to(device)

            outputs = model(images)
            _, predicted_class = torch.max(outputs, 1)
            preds_arr.append(predicted_class.cpu().numpy())
            labels_arr.append(labels)

            #print(f"Labels: {labels}")
            #print(f"Preds: {predicted_class}")
    preds_arr = np.concatenate(preds_arr)
    labels_arr = np.concatenate(labels_arr)
    #print(preds_arr)
    #print(len(preds_arr))
    return preds_arr, labels_arr