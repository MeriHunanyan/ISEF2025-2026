import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("/home/merih/ISEF2025-2026/data/raw/BBBP.csv")
subset_sizes = [50, 100, 200, 500]

def stratsample(x_train, y_train, size):
    np.random.seed(42)
    
    class_0i = np.where(y_train == 0)[0]
    class_1i = np.where(y_train == 1)[0]

    n_class0 = int(size * (len(class_0i) / len(y_train)))
    n_class1 = n_samples = n_class0
    print(class_0i)
    sample_0 = np.random.choice(class_0i, size = n_class0, replace = False)
    sample_1 = np.random.choice(class_1i, size = n_class1, replace = False)

    stratified_sample = np.concatenate([sample_0, sample_1])
    np.random.shuffle(stratified_sample)
    return x_train[stratified_sample], y_train[stratified_sample]
    


#print(dataset.describe(include = 'all')
X = dataset['smiles'].values
Y = dataset['p_np'].values
#test
X_trainval, X_test, Y_trainval, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42)

X_train, X_val, Y_train, Y_val = train_test_split(
    X_trainval, Y_trainval, test_size = 0.1, stratify=Y_trainval, random_state=42)

np.save("../data/processed/x_test.npy", X_test)
np.save("../data/processed/Y_test.npy", Y_test)
np.save("../data/processed/x_val.npy", X_val)
np.save("../data/processed/Y_val.npy", Y_val)

for size in subset_sizes:
    x_sub, y_sub = stratsample(X_train, Y_train, size)

    np.save("../data/processed/x_train"+ str(size)+".npy", x_sub)
    np.save("../data/processed/y_train"+str(size)+".npy", y_sub)
