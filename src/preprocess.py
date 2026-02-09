# diagnostic_preprocess.py
import pandas as pd
import numpy as np
from rdkit import Chem
import deepchem as dc
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer
csv_path = "/home/merih/ISEF2025-2026/data/raw/HIV.csv"
df = pd.read_csv(csv_path)
#df['smiles'] = pd.to_numeric(df['smiles'], errors='coerce')
X_raw_series = df["smiles"]
Y_raw_series = df["HIV_active"]


# ... (Assume X_raw_series, Y_raw_series are defined and cleaned into lists) ...
valid_smiles = []
valid_labels = []
print("before filtering")
# (populate lists as before) ...
for sm, lab in zip(X_raw_series, Y_raw_series):
    if sm is None:
        print(sm+"removed")
        print("hello")
        continue
     # ensure SMILES is a string
    if not isinstance(sm, str):
        #print(sm+"removed")
        print("int")
        continue
 
    if sm[0].isdigit():
        continue
    
    mol = Chem.MolFromSmiles(str(sm))
    if mol is None:
        print("hello")
        continue

    valid_smiles.append(str(sm))
    valid_labels.append(str(lab))
#convert to fingerprints
valid_X = valid_smiles
valid_X = np.array(valid_X, dtype= str)
valid_labels = np.array(valid_labels, dtype = str)
#valid_X = valid_X.reshape(-1,1)
#valid_labels = valid_labels.reshape(-1,1)
featurizer = dc.feat.CircularFingerprint(size=2048)
valid_X = featurizer.featurize(valid_X)
print(valid_X.shape)
print(valid_labels.shape)
dataset = dc.data.NumpyDataset(valid_X, valid_labels, ids=valid_smiles)

scaffoldsplitter = dc.splits.ScaffoldSplitter()
train, valid, test = scaffoldsplitter.train_valid_test_split(dataset)
print(train.X)
#featurizer = dc.feat.CircularFingerprint(size=2048)
#train_X = featurizer.featurize(train.X)
#train_y = featurizer.featurize(train.y)
#valid_X = featurizer.featurize(valid.X)
#valid_y = featurizer.featurize(valid.y)
#test_X = featurizer.featurize(test.X)
#test_y = featurizer.featurize(test.y)
#Make dataset
#dataset = dc.data.NumpyDataset(valid_finger, valid_labels)

#split the dataset into train, valid, test
#splitter = dc.splits.RandomSplitter()
#scaffoldsplitter = dc.splits.ScaffoldSplitter()
#train, valid, test = scaffoldsplitter.train_valid_test_split(dataset)


#X_train = featurizer.featurize(trainX)
#X_valid = featurizer.featurize(valid.X)
#X_test = featurizer.featurize(test.X)

#trainX = featurizer.featurize(train.X)
#trainy = featurizer.featurize(train.y)



#ADASYN oversampling
#print("after ADASYN")
#adasyn = ADASYN(sampling_strategy='minority')


#print(train.X)
#print(train.y)
#trainX, trainy = adasyn.fit_resample(train.X, train.y)
unique, counts = np.unique(train.y, return_counts=True)
print(dict(zip(unique, counts)))

#SMOTE oversampling
#print("SMOTE oversampling")
#oversample = SMOTE()
#trainX, trainy = oversample.fit_resample(train.X, train.y)
#counter = Counter(trainy)

#unique, counts = np.unique(trainy, return_counts=True)
print(dict(zip(unique, counts)))

#getting rid of outliers
#trainX = train.X
#trainy = train.y
#print("QuantileTransformer")
#quantile = QuantileTransformer(output_distribution = 'normal')
#data_trans = quantile.fit_transform(trainX)

print("beforesetting type")
print(train.y)
y_train = train.y.astype(np.int8)
y_valid = valid.y.astype(np.int8)
y_test = test.y.astype(np.int8)
print("aftersetting type")
print(y_train)
X_train = train.X
X_val = valid.X
X_test = test.X

X_trainpos = []
X_trainneg = []
X_testpos = []
X_testneg = []
X_valpos = []
X_valneg = []

i = 0
print("entering sorting part")
while len(X_trainpos)<50:
    print(y_train[i])
    if y_train[i] == 1:
        X_trainpos.append(X_train[i])
    i+=1
i = 0
while len(X_testpos)<50:
    if y_train[i] == 1:
        X_testpos.append(X_test[i])
    i+=1
i=0
while len(X_valpos)<50:
    if y_train[i] == 1:
        X_valpos.append(X_val[i])
    i+=1
i = 0
while len(X_trainneg)<50:
    if y_train[i] == 0:
        X_trainneg.append(X_train[i])
    i+= 1
i=0
while len(X_testneg)<50:
    if y_train[i] == 0:
        X_testneg.append(X_test[i])
    i+=1
i=0
while len(X_valneg)<50:
    if y_train[i] == 0:
        X_valneg.append(X_val[i])
    i+=1
#print("end first sort")
#while len(X_testpos)<50 or l`en(X_testneg)<50:
#    if y_test[i] == 1:
#        X_testpos.append(trainX[i])
#    else:
#        X_testneg.append(trainX[i])
#    i+=1
#print("second sort")
#while len(X_valpos)<50 or len(X_valneg)<50:
#    if y_test[i] == 1:
#        X_testpos.append(trainX[i])
#    else:
#        X_valneg.append(trainX[i])
#    i+=1
#while len(X_trainpos) < 50 or len(X_trainneg) < 50 or len(X_testneg) < 50 or len(X_testpos) < 50:
#    print(i)
#    if y_train[i] == 1:
#        X_trainpos.append(trainX[i])
#        if y_test[i]==1:
#            X_testpos.append(test.X[i]) # in train it's 1 and in test it's 1
#        X_testneg.append(test.X[i]) # in test it's 0
#    X_trainneg.append(trainX[i]) # in train it's 0
#    if y_test[i] == 1:
#        X_testpos.append(test.X[i]) # in train it's 0 in test it's 1
#    else:
#        X_testneg.append(test.X[i]) # in train it's 0 and in test it's 0
#    i = i+1

print("end sort")
print(f"negcounttest: {len(X_testneg)}")
print(f"negcounttrain: {len(X_trainneg)}")
print(f"poscounttest: {len(X_testpos)}")
print(f"poscounttrain: {len(X_trainpos)}")

if len(X_testneg) >50:
    X_testneg[:] = X_testneg[:50]
if len(X_trainneg) > 50:
    X_trainneg[:] = X_trainneg[:50]
if len(X_valneg) > 50:
    X_valneg[:] = X_valneg[:50]
if len(X_testpos) > 50:
    X_testpos[:] = X_testpos[:50]
if len(X_trainpos)> 50:
    X_trainpos[:] = X_trainpos[:50]
if len(X_valpos)>50:
    X_valpos[:] = X_valpos[:50]
print(f"negcounttest: {len(X_testneg)}")
print(f"negcounttrain: {len(X_trainneg)}")
print(f"poscounttest: {len(X_testpos)}")
print(f"poscounttrain: {len(X_trainpos)}")

Xtestsampled = X_testpos + X_trainneg
Xtrainsampled = X_trainpos + X_trainneg
Xvalsampled = X_valpos + X_valneg
ytestsampled = ([1] *50) + ([0] * 50)
ytrainsampled = ([1] *50) + ([0] *50)
yvalsampled = ([1]*50) + ([0] *50)

print(f"ytestsampled:{ytestsampled}")
print(f"ytrainsampled:{ytrainsampled}")
scalar = MinMaxScaler(feature_range=(-np.pi, np.pi)) #learns min and max and then applies it to scale the data to min and max
X_train_scaled = scalar.fit_transform(Xtrainsampled)
X_val_scaled = scalar.transform(Xvalsampled)
X_test_scaled = scalar.transform(Xtestsampled)
print("before pca")
pca = PCA(n_components=4)
X_trainq = pca.fit_transform(X_train_scaled)
X_valq  = pca.transform(X_val_scaled)
X_testq  = pca.transform(X_test_scaled)

#transformer = Normalizer(norm='l2').fit(X_trainq)
#X_trainq = transformer.transform(X_trainq)
#transformer = Normalizer(norm='l2').fit(X_testq)
#X_testq = transformer.transform(X_testq)

X_train = X_trainq
X_test = X_testq
X_val = X_valq
y_train = ytrainsampled
y_test = ytestsampled
y_val = yvalsampled
#regular data for classical model
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_val_tf = tf.convert_to_tensor(X_val, dtype=tf.float32)
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype = tf.float32)
y_val_tf = tf.convert_to_tensor(y_val, dtype = tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype = tf.float32)

X_trainq_tf = tf.convert_to_tensor(X_trainq, dtype = tf.float32)
X_valq_tf = tf.convert_to_tensor(X_valq, dtype = tf.float32)
X_testq_tf = tf.convert_to_tensor(X_testq, dtype = tf.float32)
y_trainq_tf = tf.convert_to_tensor(ytrainsampled, dtype = tf.float32)
y_valq_tf = tf.convert_to_tensor(yvalsampled, dtype = tf.float32)
y_testq_tf = tf.convert_to_tensor(ytestsampled, dtype = tf.float32)
print(y_train)

print(y_train_tf)

np.save("../data/processed/X_train_tf.npy", X_train_tf)
np.save("../data/processed/y_train_tf.npy", y_train_tf)
np.save("../data/processed/X_val_tf.npy", X_val_tf)
np.save("../data/processed/y_val_tf.npy", y_val_tf)
np.save("../data/processed/X_test_tf.npy", X_test_tf)
np.save("../data/processed/y_test_tf.npy", y_test_tf)

np.save("../data/processed/X_trainq.npy", X_trainq_tf)
np.save("../data/processed/X_valq.npy", X_valq_tf)
np.save("../data/processed/X_testq.npy", X_testq_tf)
np.save("../data/processed/y_trainq_tf.npy", y_trainq_tf)
np.save("../data/processed/y_valq_tf.npy", y_valq_tf)
np.save("../data/processed/y_testq_tf.npy", y_testq_tf)

print("--------------------DONE------------------------")
"""
# 1. Convert SMILES strings to RDKit Mol objects
# This ensures you have valid molecular objects for the splitter's logic
mol_objects = [Chem.MolFromSmiles(s) for s in valid_smiles]
# Filter out any that might fail MolFromSmiles (though your loop above should catch most)
valid_mols = [m for m in mol_objects if m is not None]
# Keep corresponding labels
Y_labels_filtered = np.array([lab for m, lab in zip(mol_objects, valid_labels) if m is not None], dtype=np.float32)


# 2. Create the DeepChem dataset using the RDKit Mol objects directly
# Crucially, NumPy must store these Python objects using dtype=object
X_mols_array = np.array(valid_mols, dtype=object)
Y_labels_array = Y_labels_filtered

dataset = dc.data.NumpyDataset(X_mols_array, Y_labels_array)

print("X dtype in dataset: " + str(dataset.X.dtype))
# This will print 'object', which is what the splitter expects internally.

# 3. Perform the split
splitter = dc.splits.ScaffoldSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)

print(f"Train dataset size: {len(train_dataset)}")
# ... (rest of your code) ...













# --- load csv (adjust path if needed) ---
csv_path = "/home/merih/ISEF2025-2026/data/raw/HIV.csv"
df = pd.read_csv(csv_path)
print("Loaded CSV shape:", df.shape)
print("Columns:", df.columns.tolist())

# --- configure columns (adjust names if your file different) ---
smiles_col = "smiles"
label_col = "HIV_active"   # change to HIV_activity if that's your column

if smiles_col not in df.columns or label_col not in df.columns:
    print("ERROR: expected columns not found. Available:", df.columns.tolist())
    sys.exit(1)

# --- raw series (keep pandas for checks) ---
X_raw_series = df[smiles_col]
Y_raw_series = df[label_col]

# --- Quick type summary ---
print("SMILES column dtype:", X_raw_series.dtype)
print("LABEL column dtype:", Y_raw_series.dtype)
print("First 10 SMILES sample:", X_raw_series.head(10).tolist())
print("First 10 labels sample:", Y_raw_series.head(10).tolist())

# --- collect valid SMILES and show bad ones explicitly ---
valid_smiles = []
valid_labels = []
bad_rows = []  # will store tuples (idx, raw_value, reason)

for idx, (sm, lab) in enumerate(zip(X_raw_series, Y_raw_series)):
    # check presence
    if sm is None:
        bad_rows.append((idx, sm, lab, "None"))
        continue
    # check nan
    if isinstance(sm, float) and np.isnan(sm):
        bad_rows.append((idx, sm, lab, "NaN"))
        continue
    # require string-like
    if not isinstance(sm, str):
        if isinstance(sm, (bytes, bytearray)):
            try:
                sm = sm.decode()
            except Exception:
                bad_rows.append((idx, sm, lab, "non-decodable-bytes"))
                continue
        else:
            bad_rows.append((idx, sm, lab, f"type-{type(sm)}"))
            continue
    # RDKit parse
    mol = Chem.MolFromSmiles(sm)
    if mol is None:
        bad_rows.append((idx, sm, lab, "rdkit-failed"))
        continue
    # label type check (ensure numeric)
    if isinstance(lab, (str, bytes)):
        # try convert string "0" / "1" to numeric
        try:
            lab_num = float(lab)
        except Exception:
            bad_rows.append((idx, sm, lab, "label-non-numeric"))
            continue
    # keep valid
    valid_smiles.append(str(sm))
    valid_labels.append(lab)

# --- force proper numpy arrays with concrete dtypes ---
X_smiles = np.array(valid_smiles, dtype=str)
Y_labels = np.array(valid_labels, dtype=np.float32)

# --- create DeepChem dataset (X must be strings) ---
for i, x in enumerate(X_smiles):
    if not isinstance(x, str):
        print("BAD X:", i, x, type(x))
        sys.exit(1)
dataset = dc.data.NumpyDataset(X_smiles, Y_labels)
# --- instantiate and split (this is the critical operation) ---
splitter = dc.splits.ScaffoldSplitter()
train, valid, test = splitter.train_valid_test_split(dataset)
"""

"""
valid_smiles = []
valid_labels = []

for sm, lab in zip(X_raw_series, Y_raw_series):

    # ensure SMILES is a string
    if not isinstance(sm, str):
        continue

    mol = Chem.MolFromSmiles(str(sm))
    if mol is None:
        continue

    valid_smiles.append(str(sm))
    valid_labels.append(str(lab))
"""
"""
# Convert CLEAN arrays
X_smiles = valid_smiles
Y_labels = np.array(valid_labels, dtype=np.float32)

# DeepChem dataset
print(type(X_smiles[0]))
dataset = dc.data.NumpyDataset(X_smiles, Y_labels)
print("X dtype after NumpyDataset: " + str(dataset.X.dtype))
# Split
"""
"""
featurizer = dc.feat.CircularFingerprint(size=1024)

# 2. Use a DeepChem Loader to process the raw data and featurize it
# This handles the internal creation of the dataset correctly for splitters
loader = dc.data.DataLoader(
    tasks=["your_label_name"], # Replace with a relevant name for your label/task         # We pass SMILES directly, so no field name needed in a CSV
    featurizer=featurizer,
    # This ensures your labels (Y_labels array) are correctly integrated
)

# Create the dataset using the loader's process method
# The process method can take your lists directly
dataset = loader.process(X=valid_smiles, y=np.array(valid_labels, dtype=np.float32))


splitter = dc.splits.ScaffoldSplitter()
train, valid, test = splitter.train_valid_test_split(dataset)
"""
"""
featurizer = dc.feat.CircularFingerprint(size=2048)
X_train = featurizer.featurize(train.X)
X_valid =featurizer.featurize(valid.X)
X_test = featurizer.featurize(test.X)


y_train = train.y.astype(np.float32)
y_valid = valid.y.astype(np.float32)
y_test = test.y.astype(np.float32)

scalar = MinMaxscaler() #learns min and max and then applies it to scale the data to min and max
X_train_s = scalar.fit_transform(X_train)
X_val_s = scalar.transform(X_valid)
X_test_s = scalar.transform(X_test)
selector = SelectKBest(mutual_info_classif, k=6) # selector
X_trainq = selector.fit_transform(X_train_s) 
X_valq = selector.transform(X_val_s)
X_testq = selector.transform(X_test_s)

X_train_tf = tf.convert_to_tensor(X_train_s, dtype=tf.float32)
X_val_tf = tf.convert_to_tensor(X_val_s, dtype=tf.float32)
X_train_tf = tf.convert_to_tensor(X_train_s, dtype=tf.float32)

X_trainq_tf = tf.convert_to_tensor(X_trainq, dtype = tf.float32)
X_valq_tf = tf.convert_to_tensor(X_valq, dtype = tf.float32)
X_testq_tf = tf.convert_to_tensor(X_testq, dtype = tf.float32)

y_train_tf = tf.convert_to_tensor(y_train, dtype = tf.float32)
y_val_tf = tf.convert_to_tensor(y_val, dtype = tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype = tf.float32)

np.save("../data/processed/X_train_tf.npy", X_train_tf)
np.save("../data/processed/y_train_tf.npy", y_train_tf)
np.save("../data/processed/X_val_tf.npy", X_val_tf)
np.save("../data/processed/y_val_tf.npy", y_val_tf)
np.save("../data/processed/X_test_tf.npy", X_test_tf)
np.save("../data/processed/y_test_tf.npy", y_test_tf)
np.save("../data/processed/X_trainq.npy", X_trainq_tf)
np.save("../data/processed/X_valq.npy", X_valq_tf)
np.save("../data/processed/X_testq.npy", X_testq_tf)

print("Split success. sizes ->", len(train.X), len(valid.X), len(test.X))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import deepchem as dc
from rdkit import Chem
import tensorflow as tf
import array
print(dc.__version__)


df = pd.read_csv("/home/merih/ISEF2025-2026/data/raw/HIV.csv")



X_raw = df["smiles"].astype(str).values
Y_raw = df["HIV_active"].values

valid_smiles = []
valid_labels = []

for sm, y in zip(X_raw, Y_raw):
    if not isinstance(sm, str):
        continue  # skip anything that is not a string
    mol = Chem.MolFromSmiles(sm, sanitize = False)
    if mol is None:
        continue  # skip invalid SMILES
    if Chem.SanitizeMol(mol) is not 0:
        continue
    valid_smiles.append(sm)
    valid_labels.append(y)


dataset = dc.data.NumpyDataset(
    np.array(valid_smiles, dtype=object),
    np.array(valid_labels, dtype=np.float32)
)

splitter = dc.splits.ScaffoldSplitter()
train, valid, test = splitter.train_valid_test_split(dataset)

featurizer = dc.feat.CircularFingerprint(size=2048)
X_train = featurizer.featurize(train.X)
X_valid = featurizer.featurize(valid.X)
X_test = featurizer.featurize(test.X)

y_train = train.y.astype(np.float32)
y_valid = valid.y.astype(np.float32)
y_test = test.y.astype(np.float32)



X_raw = df["smiles"].astype(str).values
Y_raw = df["HIV_active"].values
valid_idx = []
valid_smiles = []


for sm, y in zip(X_raw, Y_raw):
    if Chem.MolFromSmiles(sm) is not None:
        valid_smiles.append(sm)
        valid_idx.append(y)





dataset = dc.data.NumpyDataset(np.array(valid_smiles, dtype=object),
                               np.array(valid_idx))


scaffoldsplitter = dc.splits.ScaffoldSplitter()
train, valid, test = scaffoldsplitter.train_valid_test_split(dataset)

featurizer = dc.feat.CircularFingerprint(size=2048)
X_train = featurizer.featurize(train.X)
X_valid = featurizer.featurize(valid.X)
X_test = featurizer.featurize(test.X)

y_train = train.y
y_valid = valid.y
y_test = test.y

X_train = train.X.astype(np.float32)
y_train = train.y.astype(np.float32).flatten()

X_val   = valid.X.astype(np.float32)
y_val   = valid.y.astype(np.float32).flatten()

X_test  = test.X.astype(np.float32)
y_test  = test.y.astype(np.float32).flatten()
scalar = MinMaxscaler() #learns min and max and then applies it to scale the data to min and max
X_train_s = scalar.fit_transform(X_train)
X_val_s = scalar.transform(X_val)
X_test_s = scalar.transform(X_test)
selector = SelectKBest(mutual_info_classif, k=6) # selector
X_trainq = selector.fit_transform(X_train_s) 
X_valq = selector.fit_transform(X_val_s)
X_testq = selector.fit_transform(X_test_s)

X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
X_val_tf = tf.convert_to_tensor(X_val_scaled, dtype=tf.float32)
X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)

X_trainq_tf = tf.convert_to_tensor(X_trainq, dtype = tf.float32)
X_valq_tf = tf.convert_to_tensor(X_valq, dtype = tf.float32)
X_testq_tf = tf.convert_to_tensor(X_testq, dtype = tf.float32)

y_train_tf = tf.convert_to_tensor(y_train, dtype = tf.float32)
y_val_tf = tf.convert_to_tensor(y_val, dtype = tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype = tf.float32)

np.save("../data/processed/X_train_tf.npy", X_train_tf)
np.save("../data/processed/y_train_tf.npy", y_train_tf)
np.save("../data/processed/X_val_tf.npy", X_val_tf)
np.save("../data/processed/y_val_tf.npy", y_val_tf)
np.save("../data/processed/X_test_tf.npy", X_test_tf)
np.save("../data/processed/y_test_tf.npy", y_test_tf)
np.save("../data/processed/X_trainq.npy", X_trainq_tf)
np.save("../data/processed/X_valq.npy", X_valq_tf)
np.save("../data/processed/X_testq", X_testq_tf)
#subset_sizes = [50, 100, 200, 500]

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
"""
