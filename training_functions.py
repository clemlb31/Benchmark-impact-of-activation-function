import torch
from torch import nn


def activation_function(mode):
    if mode == 'relu':
        return nn.ReLU()
    elif mode == 'gelu':
        return nn.GELU()
    else :
        raise ValueError("bad activation function (relu,gelu)")


def train_model(model, criterion, optimizer,  num_epochs,train_loader, val_loader=None) -> dict:
    
    
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
    print(num_epochs)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
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
        