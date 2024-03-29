import numpy as np
import torch
import matplotlib.pyplot as plt

def test_model(model, test_dataloader, device):
    print('test step')
    model.train(False)
    model.to(device)

    test_loss = []

    for i, (vframes_batch, list_of_tokens) in enumerate(test_dataloader):
        vframes_batch = vframes_batch.to(device)

        with torch.no_grad(): # на test просто прогоняем модель, не собирая grad
            output = model(vframes_batch, list_of_tokens)

            loss = model.loss
            test_loss.append(loss.item())
    
    model.train(True)
    
    return test_loss

def train_model(model, train_dataloader, val_dataloader, optimizer, device):
    print('train step')
    model.train(True)
    model.to(device)
    
    train_loss = []
    val_loss = []
    X = []
    
    temp_train_loss = []
    fig, ax = plt.subplots() # для рсиунков
    for i, (vframes_batch, list_of_tokens) in enumerate(train_dataloader):
        vframes_batch = vframes_batch.to(device)

        output = model(vframes_batch, list_of_tokens)
        
        loss = model.loss
        temp_train_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 500 == 0 and i > 0:
            print('val step')
            # сохраняем модель
            PATH = f'model_{i}.pt'
            torch.save(model.state_dict(), PATH)

            X.append(i) # координаты для прорисовки
            
            mean_loss_on_train = np.array(temp_train_loss).mean() # усредняем собранный loss на train'е
            train_loss.append(mean_loss_on_train)
            
            temp_train_loss = [] # очищяем для нового накопления
            
            temp_val_loss = test_model(model, val_dataloader, device)
            mean_loss_on_val = np.array(temp_val_loss).mean() # усредняем собранный loss на val'е
            val_loss.append(mean_loss_on_val)

            
            # печатаем и рисуем полученные значения
            print(f'mean loss on train: \t{mean_loss_on_train}')
            print(f'mean loss on val: \t{mean_loss_on_val}')
            
            ax.clear()
            ax.plot(X, train_loss, label='train')
            ax.plot(X, val_loss, label='val')
            ax.savefig(f'images/result_{i}.png')

    
    return train_loss, val_loss
