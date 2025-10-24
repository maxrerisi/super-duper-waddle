from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from data_loader import load_data
from xgboost import XGBClassifier
from scipy.stats import uniform
from scipy.stats import uniform, loguniform
from scipy.stats import geom

X_train, y_train, X_test, y_test = load_data()

base = XGBClassifier(seed=42, objective="binary:logistic")
"""
eta -> 0-1
gamma -> 0-inf
max_depth 1-inf (int)
min_child_weight 1-inf 
max_delta_step 0-inf
subsample 0-1
col_sample_bytree 0-1
col_sample_bylevel 0-1
col_sample_bynode 0-1
lambda 0-inf
alpha 0-inf
tree_method=hist
"""


# Integer distributions that favor lower values
param_dist = {
    "max_depth": geom(
        p=0.3, loc=0
    ),  # geometric distribution starting at 1, favors lower values
    "min_child_weight": geom(
        p=0.2, loc=0
    ),  # geometric distribution starting at 1, favors lower values
    # Float distributions
    "eta": uniform(0.0001, 0.99),  # uniform between 0.01 and 1.0
    "gamma": loguniform(1e-8, 1.0),  # log-uniform for wide ranges
    "subsample": uniform(0.5, 0.5),  # uniform between 0.5 and 1.0
    "colsample_bytree": uniform(0.5, 0.5),
    "colsample_bylevel": uniform(0.5, 0.5),
    "colsample_bynode": uniform(0.5, 0.5),
    "reg_lambda": loguniform(1e-8, 100),  # L2 regularization
    "reg_alpha": loguniform(1e-8, 100),  # L1 regularization
}


def auc(estim, X, y):
    return roc_auc_score(estim.predict(X), y)


rs = RandomizedSearchCV(
    base,
    n_iter=100_000,
    param_distributions=param_dist,
    cv=5,
    random_state=99,
    verbose=100,
    n_jobs=4,
    scoring=auc,
)
rs.fit(X_train, y_train)
