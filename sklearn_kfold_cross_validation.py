"""
Sklearn: K-Fold Cross Validation
"""


import numpy as np
from sklearn import base
from sklearn import metrics
from sklearn import ensemble
from sklearn import model_selection
from sklearn.datasets import load_breast_cancer

seed = 42

def cross_val_estimators(base_estimator, X, y, cv=5, random_state=None):
    estimators = []
    folds = model_selection.KFold(n_splits=cv, shuffle=True, 
                                  random_state=random_state)
    oof = np.zeros(X.shape[0])
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        train_x, train_y = X[train_idx], y[train_idx]
        valid_x, valid_y = X[valid_idx], y[valid_idx]
        estimator = base.clone(base_estimator)
        estimator.fit(train_x, train_y)
        estimators.append(estimator)
        oof[valid_idx]=estimator.predict_proba(valid_x)[:,1]
    return estimators, oof


def predict(estimators, X):
    pred = np.zeros(len(X))
    for estimator in estimators:
        pred += estimator.predict_proba(X)[:,1]/len(estimators)
    return pred


X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(X, y, test_size=0.10, random_state=seed)
print(X_train.shape, X_test.shape)


clf = ensemble.RandomForestClassifier(n_estimators=10, random_state=seed)
estimators, oof = cross_val_estimators(clf, X_train, y_train, 5, 
                                       random_state=seed)
print("Cross validation accuracy:", 
      metrics.accuracy_score(y_train, np.round(oof)))

pred = predict(estimators, X_test)
print("Test accuracy:", metrics.accuracy_score(y_test, np.round(pred)))


"""
Output:
(512, 30) (57, 30)
Cross validation accuracy: 0.9453125
Test accuracy: 0.9649122807017544
"""
