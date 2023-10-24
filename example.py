import numpy as np
import pandas as pd
from functools import partial
#import matplotlib.pyplot as plt

import time
from sklearn.covariance import empirical_covariance

from regain.covariance.graphical_lasso_ import GraphicalLasso
from regain.datasets import make_dataset
from regain.scores import log_likelihood, BIC, EBIC, EBIC_m
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from regain.utils import structure_error, error_norm
from sklearn.metrics import matthews_corrcoef
from regain.model_selection.stability_optimization import GraphicalModelStabilitySelection


Xs = []
thetas = []

combos = [(1000, 10), (100, 10), (100, 100), (10, 100)]
for c in combos:
    res = make_dataset(n_samples=c[0], n_dim_obs=c[1], n_dim_lat=0, T=1)
    Xs.append(res['data'][0])
    thetas.append(res['thetas'][0])


Xs = Xs[:2]
thetas = thetas[:2]


res = make_dataset(n_samples=20, n_dim_obs=100, n_dim_lat=0, T=1)
Xs.append(res['data'][0])
thetas.append(res['thetas'][0])


def score(estimator, X_test, score_type='likelihood'):
    test_cov = empirical_covariance(X_test, assume_centered=False)

    if score_type.lower() == 'likelihood':
        return log_likelihood(test_cov, estimator.precision_)
    elif score_type.lower() == 'bic':
        return BIC(test_cov, estimator.precision_)
    elif score_type.lower() == 'ebic':
        return EBIC(test_cov, estimator.precision_)
    elif score_type.lower() == 'ebic_m':
        return EBIC_m(test_cov, estimator.precision_)
    else:
        raise ValueError(
            'Undefined type of scores, accepted scores are [likelihood, bic, ebic or ebic_m]'
        )



def test_all(X, theta):
    train_MCC = {
        k: []
        for k in ['likelihood', 'bic', 'ebic', 'ebic_m', 'stars']
    }
    test_MCC = {
        k: []
        for k in ['likelihood', 'bic', 'ebic', 'ebic_m', 'stars']
    }
    train_scores = {
        k: []
        for k in ['likelihood', 'bic', 'ebic', 'ebic_m', 'stars']
    }
    test_scores = {
        k: []
        for k in ['likelihood', 'bic', 'ebic', 'ebic_m', 'stars']
    }
    selected_parameters = {
        k: []
        for k in ['likelihood', 'bic', 'ebic', 'ebic_m', 'stars']
    }
    train_error_norm = {
        k: []
        for k in ['likelihood', 'bic', 'ebic', 'ebic_m', 'stars']
    }
    test_error_norm = {
        k: []
        for k in ['likelihood', 'bic', 'ebic', 'ebic_m', 'stars']
    }

    times = {k: [] for k in ['likelihood', 'bic', 'ebic', 'ebic_m', 'stars']}

    est = GraphicalLasso()

    sss = ShuffleSplit(n_splits=100, test_size=0.5)
    for train, test in sss.split(X):

        for t in ['likelihood', 'bic', 'ebic', 'ebic_m']:
            scoref = partial(score, score_type=t)
            tim = time.time()
            rscv = RandomizedSearchCV(
                est, param_distributions={'alpha': np.logspace(-3, -1, 100)},
                cv=5, scoring=scoref)
            rscv.fit(X[train, :])
            times[t].append(time.time() - tim)

            selected_parameters[t].append(rscv.best_estimator_.alpha)
            train_MCC[t].append(
                matthews_corrcoef(
                    theta.ravel() != 0,
                    rscv.best_estimator_.precision_.ravel() != 0))
            train_error_norm[t].append(
                error_norm(theta, rscv.best_estimator_.precision_))
            train_scores[t].append(scoref(rscv.best_estimator_, X[train, :]))
            rscv.best_estimator_.fit(X[test, :])
            test_MCC[t].append(
                (
                    matthews_corrcoef(
                        theta.ravel() != 0,
                        rscv.best_estimator_.precision_.ravel() != 0)))
            test_error_norm[t].append(
                error_norm(theta, rscv.best_estimator_.precision_))
            test_scores[t].append(scoref(rscv.best_estimator_, X[test, :]))
        sampling_size = min(
            int(10 * np.sqrt(train.shape[0])),
            train.shape[0] - int(train.shape[0] * 0.25))
        tim = time.time()
        ss = GraphicalModelStabilitySelection(
            est, n_repetitions=10, sampling_size=sampling_size,
            param_grid={'alpha': np.linspace(1e-5, 1e3, 100)})
        ss.fit(X[train, :])
        times['stars'].append(time.time() - tim)
        selected_parameters['stars'].append(ss.best_estimator_.alpha)
        train_MCC['stars'].append(
            matthews_corrcoef(
                theta.ravel() != 0,
                ss.best_estimator_.precision_.ravel() != 0))
        train_error_norm['stars'].append(
            error_norm(theta, ss.best_estimator_.precision_))
        train_scores['stars'].append(scoref(ss.best_estimator_, X[train, :]))
        ss.best_estimator_.fit(X[test, :])
        test_MCC['stars'].append(
            (
                matthews_corrcoef(
                    theta.ravel() != 0,
                    ss.best_estimator_.precision_.ravel() != 0)))
        test_error_norm['stars'].append(
            error_norm(theta, ss.best_estimator_.precision_))
        test_scores['stars'].append(scoref(ss.best_estimator_, X[test, :]))

    result = [
        pd.DataFrame.from_dict(train_MCC),
        pd.DataFrame.from_dict(test_MCC),
        pd.DataFrame.from_dict(train_scores),
        pd.DataFrame.from_dict(test_scores),
        pd.DataFrame.from_dict(train_error_norm),
        pd.DataFrame.from_dict(test_error_norm),
        pd.DataFrame.from_dict(selected_parameters),
        pd.DataFrame.from_dict(times)
    ]
    return result


X = Xs[0]
theta = thetas[0]
res_1000_10 = test_all(X, theta)


print(res_1000_10)
