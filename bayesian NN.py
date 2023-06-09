#misc
import numpy as np
import pandas as pd
from functions import functions
from matplotlib import pyplot as plt
import time
#torch utilis
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchbnn as bnn
# TensorBoard utils
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


df=functions.read_data('H:\\Datasets\\','UK-HPI-full-file-2022-01_clean.csv')
columns=functions.get_columns(df)
X_train, X_test, y_train, y_test=functions.split_data(df, 'AveragePrice',test_size=0.2)

inputs=X_train.shape[1]
outputs=len(columns)-inputs

X_train=torch.tensor(np.array(X_train))
y_train=torch.tensor(np.array(y_train))
X_test=torch.tensor(np.array(X_test))
y_test=torch.tensor(np.array(y_test))

#print([X_train,y_train])
train = [(X_train[i],y_train[i]) for i in range(len(X_train))]
test = [(X_test[i],y_test[i]) for i in range(len(X_test))]
print('Train has {} instances '.format(len(train)))
print('Test has {} instances'.format(len(test)))

batch_size=1024
train_loader=DataLoader(train, batch_size=1024, shuffle=True, drop_last=True)
test_loader=DataLoader(test, batch_size=1024, shuffle=False, drop_last=True)

# train_iter = iter(train_loader)
# features, target = train_iter.__next__()
# print(features, target)

def return_bnn(input, output, layers):
    model = nn.Sequential()
    start=bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input, out_features=layers[0])
    end=bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=layers[-1], out_features=output)
    model.append(start)
    model.append(nn.ReLU())
    if len(layers)>1:
        for i in range(len(layers)-1):
            model.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=layers[i], out_features=layers[i + 1]))
            model.append(nn.ReLU())
    model.append(end)
    return model

model=return_bnn(inputs,outputs,[1024])
print(model)
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01
optimizer = optim.Adam(model.parameters(), lr=0.001)


def do_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(train_loader):
        features, target = data
        # print(features, target)
        pre = model(torch.tensor(features, dtype=torch.float32)) # predictions
        mse = mse_loss(torch.flatten(pre), torch.tensor(target, dtype=torch.float32))
        kl = kl_loss(model)
        cost = mse + kl_weight * kl

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        num_batches=len(train)/batch_size
        # Gather data and report
        running_loss += kl.item()
        if i % 95 == 94: #for our specific case of batches
            last_loss = running_loss / batch_size  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.



    print('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/AveragePrice{}'.format(timestamp))
epoch_number = 0

EPOCHS=50
for epoch in range(EPOCHS+1):
    do_one_epoch(epoch+1, writer )
    print('Epoch {} completed'.format(epoch))
    writer.flush()
model_path = 'model_{}'.format(timestamp)
torch.save(model.state_dict(), model_path)





