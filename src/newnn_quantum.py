import tensorflow as tf
#import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

import pennylane as qml
import numpy as np

import matplotlib.pyplot as plt

#loading data
X_train = np.load("../data/processed/X_testq.npy")
y_train = np.load("../data/processed/y_testq_tf.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(8)  # no activation or linear
])

def init_layer(x):
    qml.Squeezing(x[3], x[4], wires=0)
    qml.Squeezing(x[9], x[10], wires=1)
    qml.Beamsplitter(x[5], x[6], wires=[0,1])
    qml.Rotation(x[7], wires=0)
    qml.Rotation(x[8], wires=1)
    qml.Displacement(x[1], x[2], wires=0)
    qml.Displacement(x[11], x[12], wires=1)
    qml.Kerr(x[0], wires=0)
    qml.Kerr(x[13], wires=1)

def layer(v):
    qml.Beamsplitter(v[0], v[1], wires=[0,1])
    qml.Rotation(v[2], wires=0)
    qml.Rotation(v[3], wires=1)
    qml.Squeezing(v[4], 0.0, wires=0)
    qml.Squeezing(v[5], 0.0, wires=1)
    qml.Beamsplitter(v[6], v[7], wires=[0,1])
    qml.Rotation(v[8], wires=0)
    qml.Rotation(v[9], wires=1)
    qml.Displacement(v[10], 0.0, wires=0)
    qml.Displacement(v[11], 0.0, wires=1)
    qml.Kerr(v[12], wires=0)
    qml.Kerr(v[13], wires=1)

def init_weights(layers, modes, active_sd=0.0001, passive_sd=0.1):
    
    M = 2 + 1 + 1  # Number of interferometer parameters: beamsplitter + 2 rotations

    int1_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    s_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    int2_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    dr_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    k_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)

    weights = tf.concat([int1_weights, s_weights, int2_weights, dr_weights, k_weights], axis=1)
    weights = tf.Variable(weights)

    return weights

num_modes = 2
num_basis = 4

# select a devide 
dev = qml.device("default.qubit", wires=8) 


@qml.qnode(dev, interface="tf")
def quantum_nn(inputs, var):
    # Encode input x into quantum state
    

    # iterative quantum layers
    for v in var:
        layer(v)
    
    #return [qml.expval(qml.X(0)), qml.expval(qml.X(1))]
    return qml.probs(wires=[0, 1])




num_layers = 4

weigths = init_weights(num_layers, num_modes)
shape_tup = weigths.shape
weight_shapes = {'var': shape_tup}
qlayer = qml.qnn.KerasLayer(quantum_nn, weight_shapes, output_dim=4)
model.add(qlayer)







opt = keras.optimizers.SGD(learning_rate=0.02)
model.compile(opt, loss = 'categorical_crossentropy', metrics =['accuracy'])

hybrid = model.fit(X_train, 
                   y_train,
                   epochs = 100,
                   batch_size = 50,
                   shuffle = True)
