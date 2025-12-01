#classicalmodels
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X_trainc = np.load("../data/processed/X_train_tf.npy")
y_trainc = np.load("../data/processed/y_train_tf.npy")
print(y_trainc.dtype)
X_validc = np.load("../data/processed/X_val_tf.npy")
y_validc = np.load("../data/processed/y_val_tf.npy")

X_testc = np.load("../data/processed/X_test_tf.npy")
y_testc = np.load("../data/processed/y_test_tf.npy")
print(y_trainc)
xgb_train = xgb.DMatrix(X_trainc, y_trainc, enable_categorical=True)
xgb_test = xgb.DMatrix(X_testc, y_testc, enable_categorical=True)


params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.1,
}
n=50
model = xgb.train(params=params,dtrain=xgb_train,num_boost_round=n)

print("training finished")
preds = model.predict(xgb_test)
preds = np.round(preds)
print(type(preds))
#for i in preds:
#    print(preds)
#preds_labels = np.argmax(preds, axis=1)

print("prediction finished now start accuracy")

accuracy= accuracy_score(y_testc, preds)
print('Accuracy of the model is:', accuracy*100)














