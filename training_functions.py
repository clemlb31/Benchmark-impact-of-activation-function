
import torch
from torch import nn

def activation_function(mode):
    if mode == 'relu':
        return nn.ReLU()
    elif mode == 'gelu':
        return nn.GELU()
    else :
        raise ValueError("bad activation function (relu,gelu)")


def train_model(model, criterion, optimizer,  num_epochs,train_loader, val_loader=None, device=torch.device("cpu")) -> dict:
    
    
    history = {
    'train_loss': [],
    'val_loss': [],
    'final_test_loss': None,
    'activation_function': str(model.mode),
    # 'model_parameters': model.state_dict(),
    'training_parameters': {
        'num_epochs': num_epochs,
        'batch_size': train_loader.batch_size,
        'learning_rate': optimizer.param_groups[0]['lr'],
        }
    }   
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        if val_loader is not None:
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    
            val_epoch_loss = val_loss / len(val_loader.dataset)
        else:
            val_epoch_loss = None
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_epoch_loss)
        
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}')
    return history



def evaluate_model(model, data_loader, device=torch.device("cpu"), loss_type="cross_entropy"):
    """
    loss_type: "cross_entropy" , "mse"
    """
    model.eval()

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss(reduction="sum")
    mae_loss = nn.L1Loss(reduction="sum")

    total_cls = 0
    correct = 0
    ce_sum = 0.0

    mse_sum = 0.0
    mae_sum = 0.0
    total_reg = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            if loss_type == "cross_entropy":
                # Classification: logits + labels (indices) ou one-hot
                if outputs.dim() == 2 and targets.dim() == 1:
                    ce_sum += ce_loss(outputs, targets).item() * inputs.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total_cls += targets.size(0)
                elif outputs.dim() == 2 and targets.dim() == 2 and outputs.size(1) == targets.size(1):
                    labels = targets.argmax(dim=1)
                    ce_sum += ce_loss(outputs, labels).item() * inputs.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total_cls += labels.size(0)
                else:
                    raise ValueError(f"Shapes inattendues pour classification: {outputs.shape} vs {targets.shape}")

            elif loss_type == "mse":
                if outputs.shape != targets.shape:
                    raise ValueError(f"Shapes incompatibles pour MSE: {outputs.shape} vs {targets.shape}")
                diff = outputs - targets
                mse_sum += (diff ** 2).sum().item()
                mae_sum += (outputs - targets).abs().sum().item()
                total_reg += targets.numel()

            else:
                raise ValueError(f"loss_type inconnu: {loss_type}")

    if loss_type == "cross_entropy":
        return {
            "accuracy": correct / total_cls if total_cls else 0.0,
            "cross_entropy": ce_sum / total_cls if total_cls else 0.0,
            "total_samples": total_cls,
        }
    elif loss_type == "mse":
        return {
            "mse_mean": mse_sum / total_reg if total_reg else 0.0,
            "mae_mean": mae_sum / total_reg if total_reg else 0.0,
            "total_elements": total_reg,
        }
    else:
        raise ValueError(f"loss_type inconnu: {loss_type}")