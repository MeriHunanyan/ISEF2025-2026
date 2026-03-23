#quantum data collection code"
import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot
params = np.load("../models/quantum_params.npy", allow_pickle=True)
print("Quantum model parameters loaded.")

#metadata = np.load("quantum_model_metadata.npy", allow_pickle=True).item()
#print(metadata)

def accuracy(params, X, Y):
    predictions = [variational_circuit(params, x) for x in X]
    predictions = np.sign(np.stack(predictions))
    return np.mean(predictions==Y)
X_test = np.load("../data/processed/X_trainq.npy")
y_test = np.load("../data/processed/y_trainq_tf.npy")
y_test  = 2 * y_test  - 1
n_qubits = params.shape[1]

dev = qml.device("default.qubit", wires=n_qubits)

H = qml.Hamiltonian(
        coeffs = [1/n_qubits] * n_qubits,
        observables=[qml.PauliZ(i) for i in range(n_qubits)]
)
#X_train = X_train / np.max(np.abs(X))
#X_train = np.pi * X_train
@qml.qnode(dev)
def feature_encoding(finger):
    # 'fingerprints' is a 2D array/tensor of shape (N_samples, 4)
    #qml.AngleEmbedding(features=finger, wires=range(n_qubits), rotation='X')
    qml.AngleEmbedding(features=finger, wires=range(n_qubits), rotation='Y')
    return [qml.expval(qml.PauliZ(i)) for i in range(6)]

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
    return qml.expval(H)

def predict_raw(params, X):
    return np.array([variational_circuit(params, x) for x in X])

def predict_label(raw_preds):
    return np.where(raw_preds >= 0, 1, -1)

raw_preds = predict_raw(params, X_test)
y_pred = predict_label(raw_preds)
y_test_auc = (y_test + 1) // 2   # {-1,1} → {0,1}
@qml.qnode(dev)
def my_custom_ansatz(weights):
    for layer_params in params:
        for i, wire_params in enumerate(layer_params):
            qml.RX(wire_params[0], wires=i)
            qml.RY(wire_params[1], wires=i)
            qml.RZ(wire_params[2], wires=i)

        # Entangle the qubits with CNOT gates
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])


#y_pred = accuracy(params, X_test, y_test)
print(y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test_auc, raw_preds)
accuracy = accuracy(params, X_test, y_test)
print(qml.draw(my_custom_ansatz)(params))
#print(draw.mpl(variational_circuit(params, X_test)))
print(f1)
print(auc)
print(accuracy)
