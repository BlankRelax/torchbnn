import numpy as np
import pandas as pd
from functions import functions
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchbnn as bnn
from matplotlib import pyplot as plt
import time


df=functions.read_data('H:\\Datasets\\','UK-HPI-full-file-2022-01_clean.csv')
columns=functions.get_columns(df)
X_train, X_test, y_train, y_test=functions.split_data(df, 'AveragePrice',test_size=0.2)

X_train=torch.tensor(np.array(X_train))
y_train=torch.tensor(np.array(y_train))
X_test=torch.tensor(np.array(X_test))
y_test=torch.tensor(np.array(y_test))

#print([X_train,y_train])
train = [(X_train[i],y_train[i]) for i in range(len(X_train))]
test = [(X_test[i],y_test[i]) for i in range(len(X_test))]
print('Train has {} instances '.format(len(train)))
print('Test has {} instances'.format(len(test)))


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

model=return_bnn(6,1,[1024])
print(model)
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01
optimizer = optim.Adam(model.parameters(), lr=0.001)


def do_one_epoch():
    i = 1
    for i, data in enumerate(train_loader):
        features, target = data
        # print(features, target)
        tic = time.perf_counter()
        pre = model(torch.tensor(features, dtype=torch.float32)) # predictions
        mse = mse_loss(torch.flatten(pre), torch.tensor(target, dtype=torch.float32))
        kl = kl_loss(model)
        cost = mse + kl_weight * kl

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        toc = time.perf_counter()
        #print("Batch " + str(i) + " complete in " + str(round(toc - tic, 3)) + " seconds")
        i += 1

    print('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))
EPOCHS=100
for epoch in range(EPOCHS+1):
    do_one_epoch()
    print('Epoch {} completed'.format(epoch))





models_result = np.array([model(torch.tensor(X_test,dtype=torch.float32)).data.numpy() for k in range(100)])
models_result = models_result[:,:,0]
models_result = models_result.T
mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
std_values = np.array([models_result[i].std() for i in range(len(models_result))])

plt.scatter(X_test[:,4], mean_values, label='Prediction')
plt.scatter(X_test[:,4], y_test, label='Ground Truth')
plt.legend()
plt.show()

# plt.figure(figsize=(10,8))
# plt.plot(X_test.data.numpy(),mean_values,color='navy',lw=3,label='Predicted Mean Model')
# plt.fill_between(X_test.data.numpy().T[0],mean_values-3.0*std_values,mean_values+3.0*std_values,alpha=0.2,color='navy',label='99.7% confidence interval')
# #plt.plot(x_test.data.numpy(),mean_values,color='darkorange')
# plt.plot(X_test.data.numpy(),y_test.data.numpy(),'.',color='darkorange',markersize=4,label='Test set')
# #plt.plot(X_test.data.numpy(),clean_target(X_test).data.numpy(),color='green',markersize=4,label='Target function')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')