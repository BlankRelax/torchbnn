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
train_df=pd.DataFrame(X_train, columns=columns[1:])
train_df['AveragePrice']=y_train
test_df=pd.DataFrame(X_test, columns=columns[1:])
test_df['AveragePrice']=y_test

df=DataLoader(df,batch_size=32)
train_loader=DataLoader(train_df, batch_size=32, shuffle=True)
test_loader=DataLoader(test_df, batch_size=32, shuffle=False)
print('Train has {} instances: '.format(len(train_df)))
print('Test has {} instances'.format(len(test_df)))

train_iter = iter(train_loader)
X_train, y_train=train_iter.__next__()
print(X_train)



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



# model = nn.Sequential(
#     bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=6, out_features=1024),
#     nn.ReLU(),
#     bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1024, out_features=1),
# )

model=return_bnn(6,1,[1024])
print(model)
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01
optimizer = optim.Adam(model.parameters(), lr=0.001)

# i=1
# for step in range(3000):
#     tic=time.perf_counter()
#     pre = model(torch.tensor(X_train,dtype=torch.float32))
#     mse = mse_loss(torch.flatten(pre), torch.tensor(y_train,dtype=torch.float32))
#     kl = kl_loss(model)
#     cost = mse + kl_weight * kl
#
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#     toc=time.perf_counter()
#     print("Run "+str(i)+" complete in "+str(round(toc - tic,3))+" seconds")
#     i+=1
#
# print('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))
#
# models_result = np.array([model(torch.tensor(X_test,dtype=torch.float32)).data.numpy() for k in range(10000)])
# models_result = models_result[:,:,0]
# models_result = models_result.T
# mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
# std_values = np.array([models_result[i].std() for i in range(len(models_result))])
#
# plt.scatter(X_test[:,4], mean_values, label='Prediction')
# plt.scatter(X_test[:,4], y_test, label='Ground Truth')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10,8))
# plt.plot(X_test.data.numpy(),mean_values,color='navy',lw=3,label='Predicted Mean Model')
# plt.fill_between(X_test.data.numpy().T[0],mean_values-3.0*std_values,mean_values+3.0*std_values,alpha=0.2,color='navy',label='99.7% confidence interval')
# #plt.plot(x_test.data.numpy(),mean_values,color='darkorange')
# plt.plot(X_test.data.numpy(),y_test.data.numpy(),'.',color='darkorange',markersize=4,label='Test set')
# #plt.plot(X_test.data.numpy(),clean_target(X_test).data.numpy(),color='green',markersize=4,label='Target function')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')