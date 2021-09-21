#%% fit a logistic regression model on an imbalanced classification dataset
from numpy import mean
import pandas as pd


from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

#%% read data
df = pd.read_csv("../data/creditcard.zip", index_col=0)
# df.head(2)
# df.shape
# df.columns

#%% create training data and label
X = df.drop(["Class"], axis = 1)
X.columns
X.shape

y = df["Class"]
y.shape

#%% define model
weights = {0:0.01, 1:1.0} # note the weights assigned to loss function, whereby positive cases have a proportionally higher contribution to loss

def LogReg_Imbalanced (solver, class_weight):
    model = LogisticRegression(solver, class_weight)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# print('Mean ROC AUC: %.3f' % mean(scores)) # Mean ROC AUC: 0.917

#%% run LogReg_Imbalanced with and without weighting 

LogReg_Imbalanced (solver="lbfgs", class_weight=None) 
# LogReg_Imbalanced (solver="lbfgs", class_weight=weights) 
 
#
  