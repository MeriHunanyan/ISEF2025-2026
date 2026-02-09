import tensorflow as tf
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pennylane import math
#loading the data
X_train = np.load("../data/processed/X_testq.npy")
y_train = np.load("../data/processed/y_testq_tf.npy")
X_test = np.load("../data/processed/X_trainq.npy")
y_test = np.load("../data/processed/y_trainq_tf.npy")

#convert labels to -1,1
y_train = 2 * y_train - 1
y_test  = 2 * y_test  - 1


n_qubits = 8
num_layers = 3


dev = qml.device("default.qubit", wires=n_qubits)

#classicla feature extractor
num_features = X_train.shape[1]
classical_nn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_features,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(8)  # no activation or linear
])
Xtrain_sampled = []
for x in X_train:
    x = x.reshape(-1,8)
    features = classical_nn(x)
    Xtrain_sampled.append(features)
X_train = Xtrain_sampled
@qml.qnode(dev)
def feature_encoding(finger):
    # 'fingerprints' is a 2D array/tensor of shape (N_samples, 4)
    qml.AngleEmbedding(features=finger, wires=range(n_qubits), rotation='X')
    qml.AngleEmbedding(features=finger, wires=range(n_qubits), rotation='Y')
    return [qml.expval(qml.PauliZ(i)) for i in range(8)]

@qml.qnode(dev, interface='autograd')
def variational_circuit(params, x):
    # Encode features into qubits
    #print(X_train.shape)
    #x = x.reshape(-1,8)
    #print("before classical_nn")
    #x = classical_nn(x)
    ##print(X_train.shape)
    #x = qml.numpy(x)
    #x = np.array(x)
    #print("before featureencoding")
    #x = x.reshape(1,-8)
    feature_encoding(x)
    #print("layer_params")
    # params shape: (num_layers, n_qubits, 3)
    # Each qubit in each layer has 3 parameters for RX, RY, and RZ rotations
    for layer_params in params:
        for i, wire_params in enumerate(layer_params):
            qml.RX(wire_params[0], wires=i)
            qml.RY(wire_params[1], wires=i)
            qml.RZ(wire_params[2], wires=i)

        # Entangle the qubits with CNOT gates
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])

    # Measure expectation value of Pauli-Z on the first qubit
    return qml.expval(qml.PauliZ(0))



#@qml.qnode(dev, interface='autograd')
#def quantum_model(params, x):
#    return variational_circuit(params, x)

def loss(params, X, Y):
    preds = np.stack([variational_circuit(params, x) for x in X])
    return qml.math.mean(qml.math.log(1.0 + qml.math.exp(1.0-Y*preds))) # qml.math.norm

#def loss(params, X, Y):
#    predictions = [variational_circuit(params, x) for x in X]
#    predictions = np.stack(predictions)  #stack
#    # Mean Squared Error: (fθ(x) - y)^2 averaged over the dataset
#    return np.mean((predictions - Y)**2)

def accuracy(params, X, Y):
    predictions = [variational_circuit(params, x) for x in X]
    predictions = np.sign(np.stack(predictions))
    return np.mean(predictions == Y)
#def F1
num_layers = 3
params = np.array((0.00001 * np.random.randn(num_layers, n_qubits, 3)), requires_grad=True)

#opt = qml.GradientDescentOptimizer(stepsize=0.01)
opt = qml.AdamOptimizer(stepsize=0.05)
num_epochs = 150

epoch_list = []
loss_list = []
train_acc_list = []
test_acc_list = []

#print("Unique train labels:", np.unique(y_train))
#print("Unique test labels:", np.unique(y_test))
#test_preds = [variational_circuit(params, x) for x in X_test[:10]]
#print("Raw predictions:", test_preds)
#print("Signed predictions:", np.sign(test_preds))
#print("Starting training")
for epoch in range(num_epochs):
    # Update parameters by taking a gradient step on the loss w.r.t. params
    print("before .steo")
    #print(X_train.shape)
    #X_train = X_train.reshape(-1,8)
    X_train = np.array(X_train)
    params = opt.step(lambda p: loss(p, X=X_train, Y=y_train), params)
    print("after .step")
    current_loss = loss(params, X_train, y_train)
    train_acc = accuracy(params, X_train, y_train)
    test_acc = accuracy(params, X_test, y_test)

    epoch_list.append(epoch+1)
    loss_list.append(current_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    current_loss = np.mean(current_loss)
    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {current_loss:.4f}, "
          f"Train Acc = {train_acc:.2f}, Test Acc = {test_acc:.2f}")
#metadata = {
#    "n_qubits": n_qubits,
#    "num_layers": num_layers,
#    "optimizer": "Adam",
#    "stepsize": 0.03,
#    "embedding": "AngleEmbedding-Y",
#    "entanglement": "ring",
#    "loss": "hinge"
#}

#np.save("quantum_model_metadata.npy", metadata)
np.save("../models/quantum_params.npy", params)
print("Quantum model parameters and medadata saved.")



"""
def feature_encoding(finger):
    # 'fingerprints' is a 2D array/tensor of shape (N_samples, 4)
    qml.AngleEmbedding(features=finger, wires=range(n_qubits), rotation='X')
    #qml.AngleEmbedding(features=finger, wires=range(n_qubits), rotation='Y')
    #return [qml.expval(qml.PauliZ(i)) for i in range(8)]


@qml.qnode(dev, interface='autograd')
def variational_circuit(params, x):
    # Encode features into qubits
    feature_encoding(x)

    # params shape: (num_layers, n_qubits, 3)
    # Each qubit in each layer has 3 parameters for RX, RY, and RZ rotations
    for layer_params in params:
        for i, wire_params in enumerate(layer_params):
            qml.RX(wire_params[0], wires=i)
            qml.RY(wire_params[1], wires=i)
            qml.RZ(wire_params[2], wires=i)

        # Entangle the qubits with CNOT gates
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])

    # Measure expectation value of Pauli-Z on the first qubit
    return np.real(qml.expval(qml.PauliZ(0)))



#@qml.qnode(dev, interface='autograd')
#def quantum_model(params, x):
#    return variational_circuit(params, x)

def loss(params, X, Y):
    #preds = [variational_circuit(params, x) for x in X]
    preds = variational_circuit(params, x=X)
    return (np.maximum(0.0, 1.0 - Y * preds))

# Accuracy
def accuracy(params, X, Y):
    preds = np.array([variational_circuit(params, x) for x in X], dtype=float)
    return np.mean(np.sign(preds) == Y)

#def F1
num_layers = 3
params = qml.numpy.array(0.1 * np.random.randn(num_layers, n_qubits, 3))

#opt = qml.GradientDescentOptimizer(stepsize=0.01)
opt = qml.AdamOptimizer(stepsize=0.05)
num_epochs = 150

epoch_list = []
loss_list = []
train_acc_list = []
test_acc_list = []

#
X_train_q = qml.numpy.array(classical_nn(X_train).numpy(), dtype=float)
X_test_q  = qml.numpy.array(classical_nn(X_test).numpy(), dtype=float)
y_train_q = qml.numpy.array(y_train, dtype=float)
y_test_q  = qml.numpy.array(y_test, dtype=float)
total_loss = 0
#print(np.array(newfeatures).shape())
for epoch in range(num_epochs):
    # Update parameters by taking a gradient step on the loss w.r.t. params
    print("before .steo")
    for X, y in zip(X_train_q, y_train_q):
        X = nn_quantum(
        params = opt.step(loss, params, X=X, Y=y)
        print("after .step")
        total_loss += loss(params, X=X, Y=y)
    
    avg_loss = total_loss/len(X_train_q)
    train_acc = accuracy(params, X=X_train_q, Y=y_train_q)
    test_acc = accuracy(params, X=X_test_q, Y=y_test_q)

    #epoch_list.append(epoch+1)
    #loss_list.append(avg_loss)
    #train_acc_list.append(train_acc)
    #test_acc_list.append(test_acc)

    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, "
          f"Train Acc = {train_acc:.2f}, Test Acc = test_acc")


#metadata = {
#    "n_qubits": n_qubits,
#    "num_layers": num_layers,
#    "optimizer": "Adam",
#    "stepsize": 0.03,
#    "embedding": "AngleEmbedding-Y",
#    "entanglement": "ring",
#    "loss": "hinge"
#}

#np.save("quantum_model_metadata.npy", metadata)
np.save("../models/quantum_params.npy", params)
print("Quantum model parameters and medadata saved.")
"""
