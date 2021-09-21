#%% weighted logistic regression model on an imbalanced classification dataset
from numpy import mean
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
#%%  read data
df = pd.read_csv("../data/creditcard.zip", index_col=0)

#%% Create training data and label
X = df.drop(["Class"], axis = 1)
X.columns
X.shape

y = df["Class"]
y.shape
#%% define model
weights = {0:0.01, 1:1.0} # note the weights assigned to loss function, whereby positive cases have a proportionally higher contribution to loss
model = LogisticRegression(solver='lbfgs', class_weight=weights)

#%% define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#%% evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

#%% summarize performance
print('Mean ROC AUC: %.3f' % mean(scores)) # Mean ROC AUC = 0.957 > 0.917 
# %%
