# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:47:42 2022
@author: srpv

"""

import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
#%%

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        print('Learning rate =')
        print(param_group['lr'])
        return param_group['lr']


#%%
def run_training_loop(model,train_loader,valid_loader,n_epochs,lr,device):

    Learning_rate=[]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scheduler = StepLR(optimizer, step_size = 20, gamma= 0.50 )

    # Use the cross entropy loss function
    loss_fn = nn.CrossEntropyLoss()

    # store metrics
    train_loss_history = np.zeros([n_epochs, 1])
    valid_accuracy_history = np.zeros([n_epochs, 1])
    valid_loss_history = np.zeros([n_epochs, 1])

    for epoch in range(n_epochs):
        learingrate_value = get_lr(optimizer)
        Learning_rate.append(learingrate_value)
        # Some layers, such as Dropout, behave differently during training
        model.train()
        train_loss = 0
        scheduler.step()
        
        for batch_idx, (data, target) in enumerate(train_loader):

            # Erase accumulated gradients
            
            data,target = data.to(device,dtype=torch.float),target.to(device,dtype=torch.long)
            
            
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = loss_fn(output, target)
            train_loss += loss.item()

            if batch_idx%10 == 0:
                print('[%d  %d] loss: %.4f'% (epoch+1,batch_idx+1,loss))
            # Backward pass
            loss.backward()
        
            # Weight update
            optimizer.step()

        train_loss_history[epoch] = train_loss / len(train_loader.dataset)

        # Track loss each epoch
        print('Train Epoch: %d  Average loss: %.4f' %
              (epoch + 1,  train_loss_history[epoch]))

        # Putting layers like Dropout into evaluation mode
        model.eval()

        valid_loss = 0
        correct = 0

        # Turning off automatic differentiation
        with torch.no_grad():
            for data, target in valid_loader:
                data,target = data.to(device,dtype=torch.float),target.to(device,dtype=torch.long)
                output = model(data)
                valid_loss += loss_fn(output, target).item()  # Sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max class score
                correct += pred.eq(target.view_as(pred)).sum().item()

        valid_loss_history[epoch] = valid_loss / len(valid_loader.dataset)
        valid_accuracy_history[epoch] = correct / len(valid_loader.dataset)

        print('Valid set: Average loss: %.4f, Accuracy: %d/%d (%.4f)\n' %
              (valid_loss_history[epoch], correct, len(valid_loader.dataset),
              100. * valid_accuracy_history[epoch]))
    
    return model, train_loss_history, valid_loss_history, valid_accuracy_history,Learning_rate

#%%
def test_performance(model,valid_loader,device):
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 

    # Putting layers like Dropout into evaluation mode
    model.eval()
    # Use the cross entropy loss function
    loss_fn = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0

    # Turning off automatic differentiation
    with torch.no_grad():
        for data, target in valid_loader:
            data,target = data.to(device,dtype=torch.float),target.to(device,dtype=torch.long)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max class score
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(valid_loader.dataset)
    test_accuracy = correct / len(valid_loader.dataset)

    print('Test set: Average loss: %.4f, Accuracy: %d/%d (%.4f)' %
          (test_loss, correct, len(valid_loader.dataset),
          100. * test_accuracy))
    return test_loss, test_accuracy
#%%
