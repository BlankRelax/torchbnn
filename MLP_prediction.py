import numpy as np
import tensorflow as tf
from functions import functions
from functions import createnn

df=functions.read_data('H:\\Datasets\\','UK-HPI-full-file-2022-01_clean.csv')
columns=functions.get_columns(df)
print(columns)
X_train, X_test, y_train, y_test=functions.split_data(df, 'AveragePrice',test_size=0.2)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

print('X_train shape is:', X_train.shape)
print('y_test shape is: ', y_test.shape)

print(X_test[:,4].shape)

m = createnn.NN_seq(1028, 'LR', 6)
m.add_dense([1028,1028,512,512,256, 256,128,128,64,64,32,16], 'LR')
m.add_output_layer(1)
m.printmodel()
m.model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
model = m.model
history=model.fit(X_train, y_train, batch_size=3000,epochs=200, validation_split=0.2)
createnn.plot_history(history)
model.evaluate(X_test, y_test)
y_hat=model.predict(X_test)
createnn.plot_accuracy(X_test[:,1], y_test=y_test, y_hat=y_hat)