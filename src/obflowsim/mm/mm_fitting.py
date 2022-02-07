import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# explicitly require this experimental feature prior to v1.0 of sklearn
# from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingRegressor


from qng import qng
from obnetwork import ErlangcEstimator, LoadEstimator, SqrtLoadEstimator, CondMeanWaitLDREstimator


def crossval_summarize_mm(scenario, unit, measure, X, y, flavor='lm',
                          scoring=('neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'r2'),
                          scale=False, fit_intercept=True, n_splits=5, kfold_shuffle=True, kfold_random_state=4,
                          return_train_score=True, return_estimator=True,
                          lasso_alpha=1.0, lasso_max_iter=1000, nn_max_iter=3000,
                          rf_random_state=0, rf_criterion='absolute_error', rf_min_samples_split=10,
                          col_idx_arate=None, col_idx_meansvctime=None, col_idx_numservers=None,
                          col_idx_cv2svctime=None, load_pctile=0.95):
    """

    Parameters
    ----------

    load_pctile
    measure
    unit
    scenario
    X
    y
    flavor
    scoring
    scale
    fit_intercept
    n_splits
    kfold_shuffle
    kfold_random_state
    return_train_score
    return_estimator
    lasso_alpha
    lasso_max_iter
    rf_random_state
    rf_criterion
    rf_min_samples_split
    col_idx_numservers
    col_idx_arate
    col_idx_meansvctime

    Returns
    -------

    """

    plt.ioff()

    # Create name label lists
    partitions = ['test', 'train']
    metric_names = [f"{p}_{s}" for s in scoring for p in partitions]
    var_names = X.columns.to_list()

    flavors_nocv = ['erlangc', 'load', 'sqrtload', 'condmeanwaitldr']

    flavors_w_coeffs = ['lm', 'lasso', 'lassocv', 'poly']
    # Only need mappings for flavors in flavors_w_coeffs
    flavor_estimator = {'lm': 'linearregression',
                        'lasso': 'lasso',
                        'lassocv': 'lassocv',
                        'poly': 'linearregression',
                        'spline': 'linearregression',
                        }

    steps = []
    if scale:
        steps.append(StandardScaler())

    # Create flavor specific pipelines
    if flavor == 'lm':
        steps.extend([LinearRegression(fit_intercept=fit_intercept)])
    elif flavor == 'lasso':
        steps.extend([Lasso(alpha=lasso_alpha, fit_intercept=fit_intercept, max_iter=lasso_max_iter)])
    elif flavor == 'lassocv':
        steps.extend([LassoCV(fit_intercept=fit_intercept, max_iter=lasso_max_iter)])
    elif flavor == 'rf':
        steps.extend([RandomForestRegressor(criterion=rf_criterion, oob_score=True,
                                            min_samples_split=rf_min_samples_split, random_state=rf_random_state)])
    elif flavor == 'svr':
        steps.extend([SVR()])
    elif flavor == 'nn':
        steps.extend([MLPRegressor(max_iter=3000)])
    elif flavor == 'hgbr':
        steps.extend([HistGradientBoostingRegressor(max_iter=3000)])
    elif flavor == 'poly':
        steps.extend([PolynomialFeatures(2), LinearRegression(fit_intercept=fit_intercept)])
    elif flavor == 'erlangc':
        steps.extend([ErlangcEstimator(col_idx_arate=col_idx_arate, col_idx_meansvctime=col_idx_meansvctime,
                                       col_idx_numservers=col_idx_numservers)])
    elif flavor == 'condmeanwaitldr':
        steps.extend([CondMeanWaitLDREstimator(col_idx_arate=col_idx_arate, col_idx_meansvctime=col_idx_meansvctime,
                                       col_idx_numservers=col_idx_numservers, col_idx_cv2svctime=col_idx_cv2svctime)])
    elif flavor == 'load':
        steps.extend([LoadEstimator(col_idx_arate=col_idx_arate, col_idx_meansvctime=col_idx_meansvctime)])
    elif flavor == 'sqrtload':
        steps.extend([SqrtLoadEstimator(col_idx_arate=col_idx_arate, col_idx_meansvctime=col_idx_meansvctime,
                                        pctile=load_pctile)])

    else:
        raise ValueError(f"Unknown flavor: {flavor}")

    model = make_pipeline(*steps)
    model_final = make_pipeline(*steps)

    # Run the cross validation model fitting and testing
    if not kfold_shuffle:
        kfold_random_state = None

    cv_iterator = KFold(n_splits=n_splits, shuffle=kfold_shuffle, random_state=kfold_random_state)
    scores = cross_validate(model, X, y, scoring=scoring,
                            cv=cv_iterator,
                            return_train_score=return_train_score, return_estimator=return_estimator)

    model_final.fit(X, y)

    # Extract coefficients for relevant flavors
    if flavor in flavors_w_coeffs:
        coeffs = [list(estimator.named_steps[flavor_estimator[flavor]].coef_) for estimator in scores['estimator']]
        intercept = [estimator.named_steps[flavor_estimator[flavor]].intercept_ for estimator in scores['estimator']]
    # Trying to get feature names for poly
    if flavor == 'poly':
        poly_features = [list(estimator.named_steps['polynomialfeatures'].get_feature_names_out(var_names))
                         for estimator in scores['estimator']][0]

    # Extract alphas for lassocv
    alphas = []
    if flavor == 'lassocv':
        alphas = [estimator.named_steps['lassocv'].alpha_ for estimator in scores['estimator']]

    # Extract metrics
    metrics_raw = {metric: scores[metric] for metric in metric_names}
    metrics = dict()
    for key, val in metrics_raw.items():
        if 'neg_' in key:
            new_key = key.replace('neg_', '')
            metrics[new_key] = -1 * val
        else:
            new_key = key
            metrics[new_key] = val

    metrics_df = pd.DataFrame(metrics)

    # Create predictions
    if flavor not in flavors_nocv:
        predictions = cross_val_predict(model, X, y, cv=cv_iterator)
    else:
        predictions = model_final.predict(X)

    residuals = predictions - y

    # Create scatter of actual vs predicted
    fig_scatter = prediction_scatter(y, predictions, f"{scenario} - cross_val_predict")

    # Create flavor specific results dictionaries (e.g. rf doesn't have coeffs)
    if flavor in flavors_w_coeffs:
        # If poly, need to construct var_names
        if flavor == 'poly':
            poly_features = [list(estimator.named_steps['polynomialfeatures'].get_feature_names_out(var_names))
                             for estimator in scores['estimator']][0]

            # Change '1' to 'intercept'
            poly_features[0] = 'intercept'

            coeffs_df = pd.DataFrame(coeffs)
            x_shape = coeffs_df.shape
            # Name the columns
            coeffs_df.columns = poly_features
        else:
            coeffs_df = pd.DataFrame(coeffs, columns=var_names)
            x_shape = coeffs_df.shape
            coeffs_df['intercept'] = intercept
            # Reorder columns to get last column (intercept) to be first
            n_cols = coeffs_df.shape[1]
            new_col_order = [_ for _ in range(0, n_cols - 1)]
            new_col_order.insert(0, n_cols - 1)
            coeffs_df = coeffs_df.iloc[:, new_col_order]

        # Scaling factors
        if scale:
            scaling_factors = np.array(
                [list(estimator.named_steps['standardscaler'].scale_) for estimator in scores['estimator']])
        else:
            scaling_factors = np.ones(x_shape)

        # terms corresponding to features
        if flavor == 'poly':
            unscaled_coeffs_df = coeffs_df.iloc[:, :] / scaling_factors
        else:
            unscaled_coeffs_df1 = coeffs_df.iloc[:, 1:] / scaling_factors
            # intercept
            unscaled_coeffs_df2 = coeffs_df.iloc[:, [0]]
            # Put them together with intercept as first column
            unscaled_coeffs_df = pd.concat([unscaled_coeffs_df2, unscaled_coeffs_df1], axis=1)

        # Create coefficient plot
        fig_coeffs = coeffs_by_fold(unscaled_coeffs_df, col_wrap=5, sharey=False)

        results = {'scenario': scenario,
                   'measure': measure,
                   'flavor': flavor,
                   'unit': unit,
                   'model': model_final,
                   'cv': cv_iterator,
                   'coeffs_df': unscaled_coeffs_df,
                   'metrics_df': metrics_df,
                   'scaling': scaling_factors,
                   'scaled_coeffs_df': coeffs_df,
                   'alphas': alphas,
                   'predictions': predictions,
                   'residuals': residuals,
                   'fitplot': fig_scatter,
                   'coefplot': fig_coeffs}
    else:
        results = {'scenario': scenario,
                   'measure': measure,
                   'flavor': flavor,
                   'unit': unit,
                   'model': model_final,
                   'cv': cv_iterator,
                   'metrics_df': metrics_df,
                   'predictions': predictions,
                   'residuals': residuals,
                   'fitplot': fig_scatter}

    return results


def fit_predict_mm(scenario, unit, measure, X_train, y_train, X_test, y_test, flavor='lm',
                   scale=False, fit_intercept=True, n_splits=5,
                   lassocv_shuffle=True, lassocv_random_state=4,
                   lasso_alpha=1.0, lasso_max_iter=1000, nn_max_iter=3000,
                   rf_random_state=0, rf_criterion='absolute_error', rf_min_samples_split=10):

    var_names = X_train.columns.to_list()
    flavors_w_coeffs = ['lm', 'lasso', 'lassocv', 'poly']
    flavor_estimator = {'lm': 'linearregression',
                        'lasso': 'lasso',
                        'lassocv': 'lassocv',
                        'poly': 'linearregression',
                        'spline': 'linearregression'}

    steps = []
    if scale:
        steps.append(StandardScaler())

    # Create flavor specific pipelines
    if flavor == 'lm':
        steps.extend([LinearRegression(fit_intercept=fit_intercept)])
    elif flavor == 'lasso':
        steps.extend([Lasso(alpha=lasso_alpha, fit_intercept=fit_intercept, max_iter=lasso_max_iter)])
    elif flavor == 'lassocv':
        cv_iterator = KFold(n_splits=n_splits, shuffle=lassocv_shuffle, random_state=lassocv_random_state)
        steps.extend([LassoCV(fit_intercept=fit_intercept, cv=cv_iterator, max_iter=lasso_max_iter)])
    elif flavor == 'rf':
        steps.extend([RandomForestRegressor(criterion=rf_criterion, oob_score=True,
                                            min_samples_split=rf_min_samples_split, random_state=rf_random_state)])
    elif flavor == 'svr':
        steps.extend([SVR()])
    elif flavor == 'nn':
        steps.extend([MLPRegressor(max_iter=nn_max_iter)])
    elif flavor == 'poly':
        steps.extend([PolynomialFeatures(2), LinearRegression(fit_intercept=fit_intercept)])
    else:
        raise ValueError(f"Unknown flavor: {flavor}")

    model = make_pipeline(*steps)

    # Fit the model on training data
    model.fit(X_train, y_train)

    # Extract coefficients for relevant flavors
    if flavor in flavors_w_coeffs:
        coeffs = [model.named_steps[flavor_estimator[flavor]].coef_]
        intercept = model.named_steps[flavor_estimator[flavor]].intercept_

    # Trying to get feature names for poly
    if flavor == 'poly':
        poly_features = model.named_steps['polynomialfeatures'].get_feature_names_out(var_names)

    # Extract alpha for lassocv
    alpha = None
    if flavor == 'lassocv':
        alpha = model.named_steps['lassocv'].alpha_

    # Create predictions
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    # Compute training and test metrics
    mae_train = mean_absolute_error(y_train, predictions_train)
    mape_train = mean_absolute_percentage_error(y_train, predictions_train)

    mae_test = mean_absolute_error(y_test, predictions_test)
    mape_test = mean_absolute_percentage_error(y_test, predictions_test)

    metrics = [{'mae_train': mae_train, 'mape_train': mape_train,
                'mae_test': mae_test, 'mape_test': mape_test}]

    metrics_df = pd.DataFrame(metrics)

    min_pred = np.min(predictions_test)

    fig_scatter = prediction_scatter(y_test, predictions_test, f"{scenario} - Actual vs Prediction (test)", min_pred)

    # Create flavor specific results dictionaries (e.g. rf doesn't have coeffs)
    if flavor in flavors_w_coeffs:

        if flavor == 'poly':

            # Change '1' to 'intercept'
            poly_features[0] = 'intercept'
            coeffs_df = pd.DataFrame(coeffs)
            coeffs_df.columns = poly_features
        else:
            coeffs_df = pd.DataFrame(coeffs, columns=var_names)
            coeffs_df['intercept'] = intercept
            # Reorder columns to get last column (intercept) to be first
            n_cols = coeffs_df.shape[1]
            new_col_order = [_ for _ in range(0, n_cols - 1)]
            new_col_order.insert(0, n_cols - 1)
            coeffs_df = coeffs_df.iloc[:, new_col_order]

        # Create coefficient plot
        fig_coeffs = coeffs_by_fold(coeffs_df, col_wrap=5, sharey=False)

        results = {'scenario': scenario,
                   'measure': measure,
                   'flavor': flavor,
                   'model': model,
                   'unit': unit,
                   'coeffs_df': coeffs_df,
                   'metrics_df': metrics_df,
                   'alpha': alpha,
                   'predictions_test': predictions_test,
                   'fitplot': fig_scatter,
                   'coefplot': fig_coeffs}
    else:
        results = {'scenario': scenario,
                   'measure': measure,
                   'flavor': flavor,
                   'unit': unit,
                   'model': model,
                   'metrics_df': metrics_df,
                   'predictions_test': predictions_test,
                   'fitplot': fig_scatter}

    return results


def prediction_scatter(actual, predicted, title, ax_anchor=0):
    """
    Create scatter plot of actual vs predicted with axline included

    :param y: Actual values
    :param predictions: Predicted values
    :return: Figure
    """
    fig = Figure()
    ax = fig.add_subplot()
    ax.scatter(actual, predicted)
    ax.axline((ax_anchor - 0.1 * ax_anchor, ax_anchor - 0.1 * ax_anchor), slope=1)
    ax.set_xlabel('actual')  # Add an x-label to the axes.
    ax.set_ylabel('predicted')  # Add a y-label to the axes.
    ax.set_title(title)  # Add a title to the axes.
    return fig


def coeffs_by_fold(coeffs_df, col_wrap=5, sharey=False):
    """
    Create faced plot showing coefficient values across folds in cross-validation

    Parameters
    ----------
    coeffs_df : Dataframe
        Coefficient values (rows are folds, columns are coefficient names)
    col_wrap : int
        number of columns to wrap in faceted plot
    sharey : bool
        True means y-axis shared across subplots

    Returns
    -------
    seaborn.FacetGrid

    """

    coeffs = coeffs_df.melt(var_name='coeff', value_name='value', ignore_index=False)
    coeffs.reset_index(drop=False, inplace=True)
    coeffs.rename(columns={'index': 'fold'}, inplace=True)

    plt.ioff() # Trying to get Seaborn to stop opening plot windows
    g = sns.FacetGrid(coeffs, col="coeff", col_wrap=col_wrap, sharey=sharey)
    g.map_dataframe(sns.barplot, x="fold", y="value")
    g.set_titles('{col_name}')
    plt.close() # Trying to get Seaborn to stop opening plot windows
    return g




if __name__ == '__main__':
    pass
