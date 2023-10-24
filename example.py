import numpy as np
import pandas as pd
import time

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from regain import datasets, utils
from regain.covariance import LatentTimeGraphicalLasso


#np.random.seed(0)

# setting 1
alpha = 0.45
tau = 3
beta = 50
eta = 10

n_samples = 100
n_dim_lat = 20
T = 10
n_dim_obs = 100
data = datasets.make_dataset(
    n_samples=n_samples, n_dim_lat=n_dim_lat, n_dim_obs=n_dim_obs, T=T,
    epsilon=1e-1, proportional=True, degree=2, keep_sparsity=True,
    update_ell='l2', update_theta='l2', normalize_starting_matrices=True)
X, y = data.X, data.y



mdl = LatentTimeGraphicalLasso(
    assume_centered=0, verbose=0, rtol=1e-5, tol=1e-5, max_iter=250,
    rho=1. / np.sqrt(X.shape[0]))


# tau=[1, 3], alpha=[.45, 1], beta=[20, 50], eta=[5, 10]

param_grid = dict(tau=[3], alpha=[.45, 1], beta=[20, 50], eta=[10])
cv = StratifiedShuffleSplit(2, test_size=0.2)
ltgl = GridSearchCV(mdl, param_grid, cv=cv, verbose=1)
ltgl.fit(X, y)


print(f"Score: {utils.structure_error(data.thetas, ltgl.best_estimator_.precision_)['accuracy']}")
