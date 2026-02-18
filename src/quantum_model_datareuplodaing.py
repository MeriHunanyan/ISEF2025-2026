#
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_train = np.load("../data/processed/X_testq.npy")
y_train = np.load("../data/processed/y_testq_tf.npy")

X_test = np.load("../data/processed/X_trainq.npy")
y_test = np.load("../data/processed/y_trainq_tf.npy")

X_valid = np.load("../data/processed/X_valq.npy")
y_valid = np.load("../data/processed/y_valq_tf.npy")
# = np.load("../data/processed/X_ 
y_train = 2 * y_train - 1
y_test  = 2 * y_test  - 1
y_valid = 2 * y_valid - 1
n_qubits = 6 # change to pca values
dev = qml.device("default.qubit", wires=n_qubits)


H = qml.Hamiltonian(
    coeffs=[1/n_qubits] * n_qubits,
    observables=[qml.PauliZ(i) for i in range(n_qubits)]
)

#X_train = X_train / np.max(np.abs(X))
#X_train = np.pi * X_train
@qml.qnode(dev)
def feature_encoding(finger):
    # 'fingerprints' is a 2D array/tensor of shape (N_samples, 4)
    #qml.AngleEmbedding(features=finger, wires=range(n_qubits), rotation='X')
    qml.AngleEmbedding(features=finger, wires=range(n_qubits), rotation='Y')
    #qml.AmplitudeEmbedding(finger, wires=range(n_qubits), normalize=True)
    return [qml.expval(qml.PauliZ(i)) for i in range(6)]

@qml.qnode(dev, interface='autograd')
def variational_circuit(params, x):
    # Encode features into qubits
    x_chunks = [] 
    for i in range(0, len(x), 5):
        x_chunks.append(x[i:i+6])
    #print(len(x_chunks))
    #print("x_chunks")
    #print(x_chunks)
    # params shape: (num_layers, n_qubits, 3)
    # Each qubit in each layer has 3 parameters for RX, RY, and RZ rotations
    for i_layer, layer_params in enumerate(params):
        qml.AngleEmbedding(features=x_chunks[i_layer % len(x_chunks)], wires=range(n_qubits), rotation='Y')
        #feature_encoding(x_chunks[i_layer])
        for i, wire_params in enumerate(layer_params):
            qml.RX(wire_params[0], wires=i)
            qml.RY(wire_params[1], wires=i)
            qml.RZ(wire_params[2], wires=i)
    
       # Entangle the qubits with CNOT gates
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
    #for layer in params:
    #    qml.StronglyEntanglingLayers(layer, wires=range(n_qubits))
    #print("---------------------------AAAAAAAAAAAAAAAAAAAAHHHHHHHHHHH------------")
    # Measure expectation value of Pauli-Z on the first qubit
    #return qml.expval(qml.PauliZ(0))
    #return #sum(qml.expval(qml.PauliZ(i)) for i in range(n_qubits)) / n_qubits
    return qml.expval(H)


#@qml.qnode(dev, interface='autograd')
#def quantum_model(params, x):
#    return variational_circuit(params, x)

#def loss(params, X, Y):
#    preds = np.stack([variational_circuit(params, x) for x in X])
#    return np.mean(np.maximum(0, 1 - Y * preds))

def loss(params, X, Y, lam=5e-4):
    preds = np.array([variational_circuit(params, x) for x in X])
    reg = lam * np.sum(params**2)
    preds = np.tanh(5*preds)
    return np.mean(np.log(1 + np.exp(-Y * preds)))

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
num_layers = 4
params = 0.01 * np.random.randn(num_layers, n_qubits, 3)

#opt = qml.GradientDescentOptimizer(stepsize=0.01)
opt = qml.AdamOptimizer(stepsize=0.02)
num_epochs = 300

epoch_list = []
loss_list = []
train_acc_list = []
test_acc_list = []
val_params = []
valid_acc_list = []
#print("Unique train labels:", np.unique(y_train))
#print("Unique test labels:", np.unique(y_test))
#test_preds = [variational_circuit(params, x) for x in X_test[:10]]
#print("Raw predictions:", test_preds)
#print("Signed predictions:", np.sign(test_preds))
#print("Starting training")
for epoch in range(num_epochs):
    # Update parameters by taking a gradient step on the loss w.r.t. params
    print("before .steo")
    params = opt.step(loss, params, X=X_train, Y=y_train)
    print("after .step")
    current_loss = loss(params, X_train, y_train)
    train_acc = accuracy(params, X_train, y_train)
    test_acc = accuracy(params, X_test, y_test)
    valid_acc = accuracy(params, X_valid, y_valid)
    
    epoch_list.append(epoch)
    loss_list.append(current_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    valid_acc_list.append(valid_acc)
    val_params.append(params)
    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {current_loss:.4f}, "
          f"Train Acc = {train_acc:.2f}, Test Acc = {test_acc:.2f}, val ACC = {valid_acc:.2f}")
biggest_acc = 0
for i in range(0,len(valid_acc_list)):
    if valid_acc_list[i]>biggest_acc:
        biggest_acc = valid_acc_list[i]
        val_index = i
plt.plot(epoch_list, loss_list, 'r--')
plt.legend(['Training loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.savefig('my_plot.png', dpi=300)
plt.show()

#metadata = {
#    "n_qubits": n_qubits,
#    "num_layers": num_layers,
#    "optimizer": "Adam",
#    "stepsize": 0.03,
#    "embedding": "AngleEmbedding-Y",
#    "entanglement": "ring",
#    "loss": "hinge"
#}`
#np.save("quantum_model_metadata.npy", metadata)
np.save("../models/quantum_params.npy", val_params[val_index])
print("Quantum model parameters and medadata saved.")
"""
# cutoff
np.random.seed(0)
num_qubits = 4
num_layers = 2
weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

print("Weights:", weights_init)
print("Bias: ", bias_init)

opt = NesterovMomentumOptimizer(0.5)
batch_size = 5


sample = X_fingertrain[0]
# Number of qubits
n_qubits = 8

# Define a quantum device with 8 qubits
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def batched_circuit(finger):
    # 'fingerprints' is a 2D array/tensor of shape (N_samples, 4)
    qml.AngleEmbedding(features=finger, wires=range(8), rotation='Y')
    return [qml.expval(qml.PauliZ(i)) for i in range(8)]

# scaled_2d_array has shape (100, 4)
results = batched_circuit(X_fingertrain) 
print(results)

def layer(layer_weights):
    for wire in range(4):
        qml.Rot(*layer_weights[wire], wires=wire)

    for wires in ([0, 1], [1, 2], [2, 3], [3, 0]):
        qml.CNOT(wires)

def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)

def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc
def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2)


@qml.qnode(dev)
def circuit(weights, fingers):
    batched_circuit(fingers)

    for layer_weights in weights:
        layer(layer_weights)

    return qml.expval(qml.PauliZ(0))

X = np.array(X_fingertrain[:, :-1])
Y = np.array(y_fingertrain[:, -1])
Y = Y * 2 - 1  # shift label from {0, 1} to {-1, 1}
weights = weights_init
bias = bias_init
for it in range(100):

    # Update the weights by one optimizer step, using only a limited batch of data
    batch_index = np.random.randint(0, len(X), (batch_size,))
    X_batch = X[batch_index]
    Y_batch = Y[batch_index]
    weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch)

    # Compute accuracy
    predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X]

    current_cost = cost(weights, bias, X, Y)
    acc = accuracy(Y, predictions)

    print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")



#@qml.qnode(dev)
#def circuit(x):
#    qml.AngleEmbedding(features=x, wires=range(n_qubits), rotation="X")
#    return qml.state()

# Generate a random real vector of length n_qubits
#x = np.random.uniform(0, np.pi, (n_qubits))

# Execute the circuit to encode the vector as a quantum state
#circuit(sample)






#def layer(layer_weights):
#    for wire in range(4):
#        qml.Rot(*layer_weights[wire], wires=wire)
#
#    for wires in ([0, 1], [1, 2], [2, 3], [3, 0]):
#        qml.CNOT(wires)



#@qml.qnode(dev)





#dev = qml.device('default.qubit', wires=3)

"""

print("-------------------------Done-------------------------")
