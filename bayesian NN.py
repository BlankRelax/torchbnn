import numpy as np
from functions import functions
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
from matplotlib import pyplot as plt

df=functions.read_data('H:\\Datasets\\','UK-HPI-full-file-2022-01_clean.csv')
columns=functions.get_columns(df)
print(columns)
X_train, X_test, y_train, y_test=functions.split_data(df, 'AveragePrice',test_size=0.8)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

print('X_train shape is:', X_train.shape)
print('y_test shape is: ', y_test.shape)
print(torch.tensor(y_train,dtype=torch.float32).shape)

model = nn.Sequential(
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=6, out_features=1000),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1000, out_features=1),
)

mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01

optimizer = optim.Adam(model.parameters(), lr=0.01)

for step in range(20):
    pre = model(torch.tensor(X_train,dtype=torch.float32))
    mse = mse_loss(pre, torch.flatten(torch.tensor(y_train,dtype=torch.float32)))
    kl = kl_loss(model)
    cost = mse + kl_weight * kl

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

print('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))

models_result = np.array([model(torch.tensor(X_test,dtype=torch.float32)).data.numpy() for k in range(50)])
models_result = models_result[:,:,0]
models_result = models_result.T
mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
std_values = np.array([models_result[i].std() for i in range(len(models_result))])

plt.scatter(X_test[:,4], mean_values)
plt.scatter(X_test[:,4], y_test)
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