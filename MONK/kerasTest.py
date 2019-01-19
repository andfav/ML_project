import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from DataSet import DataSet, ModeInput
from Input import OneOfKAttribute, Input, OneOfKInput, TRInput, OneOfKTRInput
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, regularizers

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

skipRow = [1,2,3,4,5,6,7,8,9,10]
columnSkip = [1]
target = [12,13]

trSet = DataSet("Z:\Matteo\Desktop\Machine Learning\ML-CUP18-TR.csv", ",", ModeInput.TR_INPUT, target, None, skipRow, columnSkip)
trSet.restrict(0,1)
list_of_input_arrays = list()
list_of_target_arrays = list()

for i in range (0,1016):
    elem = trSet.getInputs()[i]
    list_of_input_arrays.append(elem.getInput())
    list_of_target_arrays.append(elem.getTarget())

inputs = np.array(list_of_input_arrays)
targets = np.array(list_of_target_arrays)

print("TrSet completo")

lr = 2e-5
decay = 0
momentum = 0.9
regularization = 0
units = 300

model = Sequential()
model.add(Dense(units=units, activation='softplus', input_dim=10, kernel_regularizer=regularizers.l2(regularization)))
model.add(Dense(units=2, activation='linear', kernel_regularizer=regularizers.l2(regularization)))

sgd = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=False)
model.compile(loss=euclidean_distance_loss, optimizer=sgd)
history = model.fit(inputs, targets, epochs=400000, batch_size=80, verbose=1, shuffle=True)

# Plot training & validation accuracy values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(["lr="+str(lr)+" decay="+str(decay)+" momentum="+str(momentum)+" reg="+str(regularization)+" units="+str(units)], loc='upper left')
plt.show()
