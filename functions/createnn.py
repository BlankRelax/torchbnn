import tensorflow as tf
from keras_sequential_ascii import keras2ascii #for printing the model
from matplotlib import pyplot as plt

def plot_history(history):
    x=history.history.keys()
    for i in range(0,len(x)):
        plt.plot(history.history[i])
        plt.title('training metric(epoch)')
        plt.ylabel(x[i])
        plt.xlabel('epoch')
        plt.legend(x, loc='lower right')
    plt.show()


class NN_seq:

    def __init__(self, nodes, activation, input):
        self.model = tf.keras.models.Sequential()
        self.dropoutval = 0.6
        if activation == 'LR':
            self.model.add(tf.keras.layers.Dense(nodes, activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=(input, )))



    def add_dense(self, nodes_layers, activation):
            if activation == 'LR':
                i=1
                for num in nodes_layers:
                    if i%2==0 and num>64:
                        self.model.add(tf.keras.layers.Dropout(self.dropoutval))
                    self.model.add(tf.keras.layers.Dense(num, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
                    i+=1
            else:
                i=0
                for num1 in nodes_layers:
                    self.model.add(tf.keras.layers.Dense(num1, activation[i]))

    def add_output_layer(self, nodes):
        self.model.add(tf.keras.layers.Dense(nodes))
    def printmodel(self):
        print(keras2ascii(self.model))
class CNN(NN_seq):
    def __init__(self, filters, kernal, activation, input):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(int(filters), kernel_size=kernal, activation=activation, input_shape=input))

    def add_Conv2D(self, filters, kernal, activation, pool_size, pooling):
        for i in range(len(filters)):
            self.model.add(tf.keras.layers.Conv2D(int(filters[i]), kernel_size=kernal[i], activation=activation[i]))
            if pooling[i]=='avg2d':
                self.model.add(tf.keras.layers.AveragePooling2D(pool_size[i]))
            elif pooling[i]=='max2d':
                self.model.add(tf.keras.layers.MaxPooling2D(pool_size[i]))
            elif pooling[i] == 'globmax2d':
                self.model.add(tf.keras.layers.GlobalMaxPool2D())
            else:
                print(pooling[i], "is not a supported pooling function, please check documentation")







# m = NN_seq(1028, 'LR', 14)
# m.add_dense([1028,1028,512,512,256, 256,128,128,64,64,32,16], 'LR')
# m.add_output_layer(1)
# m.printmodel()
# m = CNN(64, (3, 3), 'relu',(256,256,3))
# m.add_Conv2D([64,32,32,32,128], [(3,3), (3,3), (3,3), (3,3), (5,5)], ['relu', 'relu', 'relu', 'relu', 'relu'], [(2,2), (2,2), (2,2), (2,2)],
#              ['avg2d', 'max2d', 'max2d','max2d', 'globmax2d'])
# m.add_dense([128, 64, 32, 1], ['relu', 'relu', 'relu','sigmoid'])
# m.printmodel()