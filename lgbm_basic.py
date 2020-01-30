

"""
LightGBM: basic example using Sklearn API
"""

import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn.datasets import load_breast_cancer
from lightgbm import LGBMClassifier

seed = 42

X, y = load_breast_cancer(return_X_y=True)

X_train, X_valid, y_train, y_valid = \
    model_selection.train_test_split(X, y, test_size=0.10, random_state=seed)
print(X_train.shape, X_valid.shape)

model_params = {
    "objective": "binary",
    "n_estimators": 100,
    "num_leaves": 7,
    "min_data_in_leaf": 10,
    "learning_rate": 0.1,
    "max_depth": 3,
    "verbosity": 1,
    'metric': 'None',
    "random_state": seed
}

fit_params = {
    "eval_set": [(X_train, y_train), (X_valid, y_valid)],
    "eval_metric": 'binary_error',
    "verbose":  1,
    "early_stopping_rounds": 10
}

lgbm = LGBMClassifier(**model_params)
lgbm.fit(X_train, y_train, **fit_params)
y_pred = lgbm.predict(X_valid)

print("Valid accuracy:", metrics.accuracy_score(y_valid, y_pred))

"""
Output:

[LightGBM] [Info] Start training from score 0.485902
[1]     valid_0's binary_error: 0.380859        valid_1's binary_error: 0.298246
Training until validation scores don't improve for 10 rounds.
[2]     valid_0's binary_error: 0.359375        valid_1's binary_error: 0.280702
[3]     valid_0's binary_error: 0.0527344       valid_1's binary_error: 0.0526316
[4]     valid_0's binary_error: 0.0449219       valid_1's binary_error: 0.0526316
[5]     valid_0's binary_error: 0.0429688       valid_1's binary_error: 0.0526316
[6]     valid_0's binary_error: 0.0351562       valid_1's binary_error: 0.0526316
[7]     valid_0's binary_error: 0.0351562       valid_1's binary_error: 0.0526316
[8]     valid_0's binary_error: 0.0253906       valid_1's binary_error: 0.0526316
[9]     valid_0's binary_error: 0.0253906       valid_1's binary_error: 0.0350877
[10]    valid_0's binary_error: 0.0234375       valid_1's binary_error: 0.0350877
[11]    valid_0's binary_error: 0.0195312       valid_1's binary_error: 0.0350877
[12]    valid_0's binary_error: 0.0175781       valid_1's binary_error: 0.0350877
[13]    valid_0's binary_error: 0.0195312       valid_1's binary_error: 0.0350877
[14]    valid_0's binary_error: 0.0136719       valid_1's binary_error: 0.0175439
[15]    valid_0's binary_error: 0.0136719       valid_1's binary_error: 0.0350877
[16]    valid_0's binary_error: 0.0117188       valid_1's binary_error: 0.0175439
[17]    valid_0's binary_error: 0.0175781       valid_1's binary_error: 0.0175439
[18]    valid_0's binary_error: 0.0117188       valid_1's binary_error: 0.0175439
[19]    valid_0's binary_error: 0.0136719       valid_1's binary_error: 0.0175439
[20]    valid_0's binary_error: 0.00976562      valid_1's binary_error: 0.0175439
[21]    valid_0's binary_error: 0.00976562      valid_1's binary_error: 0.0175439
[22]    valid_0's binary_error: 0.00976562      valid_1's binary_error: 0.0175439
[23]    valid_0's binary_error: 0.0078125       valid_1's binary_error: 0.0175439
[24]    valid_0's binary_error: 0.0078125       valid_1's binary_error: 0.0175439
Early stopping, best iteration is:
[14]    valid_0's binary_error: 0.0136719       valid_1's binary_error: 0.0175439
Valid accuracy: 0.9824561403508771
"""


