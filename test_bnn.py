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


X_test=torch.tensor(np.array(X_test))
y_test=torch.tensor(np.array(y_test))



test = [(X_test[i],y_test[i]) for i in range(len(X_test))]
print('Test has {} instances'.format(len(test)))

batch_size=1024

test_loader=DataLoader(test, batch_size=1024, shuffle=False, drop_last=True)
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
model.load_state_dict(torch.load('model_20230608_133807'))

def make_predictions(num_pred,y_test_shape, plot):
    model_result_a =np.empty(shape=[num_pred, y_test_shape])
    for i in range(num_pred):
        models_result = np.array([model(torch.tensor(X_test, dtype=torch.float32)).data.numpy() for k in range(100)])
        models_result = models_result[:, :, 0]
        models_result = models_result.T
        mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
        np.append(model_result_a, mean_values)
        std_values = np.array([models_result[i].std() for i in range(len(models_result))])
        if plot:
            plt.scatter(X_test[:, 4], mean_values, label='Prediction')
            plt.scatter(X_test[:, 4], y_test, label='Ground Truth')
            plt.subplot(3, 3, i + 1)
    if plot:
        plt.legend()
        plt.show()


    return model_result_a
predictions=make_predictions(4,y_test.shape[0],True)
print(predictions.shape)


def get_model_accuracy(predictions,y_test): #takes an array of seperate predictions on the test data
    for prediction in predictions:
        delta=prediction[0]-np.array(y_test)[0]
        mse=(np.square(delta).sum()/prediction.shape[0])
        acc=delta/np.array(y_test)[0].sum()
        print('mse is {}'.format(mse))
        print('acc is {}'.format(acc))
get_model_accuracy(predictions,y_test)







# plt.figure(figsize=(10,8))
# plt.plot(X_test.data.numpy(),mean_values,color='navy',lw=3,label='Predicted Mean Model')
# plt.fill_between(X_test.data.numpy().T[0],mean_values-3.0*std_values,mean_values+3.0*std_values,alpha=0.2,color='navy',label='99.7% confidence interval')
# #plt.plot(x_test.data.numpy(),mean_values,color='darkorange')
# plt.plot(X_test.data.numpy(),y_test.data.numpy(),'.',color='darkorange',markersize=4,label='Test set')
# #plt.plot(X_test.data.numpy(),clean_target(X_test).data.numpy(),color='green',markersize=4,label='Target function')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')