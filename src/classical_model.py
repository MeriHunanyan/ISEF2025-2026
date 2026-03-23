#classicalmodels
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from numpy import arange
from numpy import argmax



def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')


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
    'max_depth': 4,
    'learning_rate': 0.05,
    'scale_pos_weight': 1, #27.85614
    'eval_metric': 'auc',
    'min_child_weight':4,
    'gamma':0.5,
    'subsample': 0.8,
    "colsample_bytree":1
}
n=300
#model = xgb.train(params, xgb_train, num_boost_round=100)
model = xgb.train(params=params,dtrain=xgb_train,num_boost_round=n)
#model = xgb.XGBClassifier(**params)


print("training finished")
preds = model.predict(xgb_test)


fpr, tpr, thresholds = roc_curve(y_validc, preds)
#optimal_idx = np.argmax(tpr - fpr)
#optimal_threshold = thresholds[optimal_idx]

# Apply the custom cutoff
#custom_preds = (preds >= optimal_threshold).astype(int)
#preds = np.round(preds)

print(type(preds))
#for i in preds:
#    print(preds)
#preds_labels = np.argmax(preds, axis=1)


#pos_preds = preds[:, 1]
#define thresholds
thresholds = arange(0, 1, 0.001)
## evaluate each threshold
scores = [f1_score(y_validc, to_labels(preds, t)) for t in thresholds]
# get best threshold
ix = argmax(scores)
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))



#fpr, tpr, thresholds = roc_curve(y_testc, preds)
preds = (preds >= 0.5).astype(int)  #0.771 thresholds[ix]
print("prediction finished now start accuracy")

accuracy= accuracy_score(y_testc, preds)
print('Accuracy of the model is:', accuracy*100)

auc_score_manual = auc(fpr, tpr)
print(f"Manual AUC Score: {auc_score_manual}")
f1 = f1_score(y_testc, preds)
print(f"f1: {f1}" )












