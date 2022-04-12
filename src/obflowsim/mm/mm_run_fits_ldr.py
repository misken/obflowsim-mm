import sys
import argparse
from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from obflowsim.mm.mm_fitting import crossval_summarize_mm
from obflowsim.mm.mm_process_fitted_models import create_cv_plots, create_coeff_plots
from obflowsim.mm.mm_process_fitted_models import create_metrics_df, create_predictions_df

# Make sure no interactive plotting happens during plot generation
plt.ioff()

UNIT = "ldr"


def fit_models(experiment, input_path, output_path, figures_path):
    """
    Fit metamodels for LDR unit

    Parameters
    ----------
    experiment : str, identifier used in folder and filenames

    Returns
    -------

    """
    unit = UNIT
    # input_path = Path(f"input/{experiment}")
    # output_path = Path(f"output/{experiment}")
    # figures_path = Path(f"output/{experiment}", "figures")

    metrics_path_filename = Path(output_path, f"{experiment}_{unit}_metrics.csv")
    pickle_path_filename = Path(output_path, f"{experiment}_{unit}_results.pkl")

    # X matrices
    X_ldr_noq = pd.read_csv(Path(input_path, f'X_ldr_noq_{experiment}.csv'), index_col=0)
    X_ldr_basicq = pd.read_csv(Path(input_path, f'X_ldr_basicq_{experiment}.csv'), index_col=0)
    X_ldr_q = pd.read_csv(Path(input_path, f'X_ldr_q_{experiment}.csv'), index_col=0)

    X_ldr_occmean_onlyq = pd.read_csv(Path(input_path, f'X_ldr_occmean_onlyq_{experiment}.csv'), index_col=0)
    X_ldr_occp95_onlyq = pd.read_csv(Path(input_path, f'X_ldr_occp95_onlyq_{experiment}.csv'), index_col=0)
    X_ldr_probblocked_onlyq = pd.read_csv(Path(input_path,
                                                   f'X_ldr_probblocked_onlyq_{experiment}.csv'),
                                              index_col=0)
    X_ldr_condmeantimeblocked_onlyq = \
        pd.read_csv(Path(input_path, f'X_ldr_condmeantimeblocked_onlyq_{experiment}.csv'), index_col=0)

    # y vectors
    y_ldr_occmean = pd.read_csv(Path(input_path, f'y_ldr_occmean_{experiment}.csv'), index_col=0).squeeze("columns")
    y_ldr_occp95 = pd.read_csv(Path(input_path, f'y_ldr_occp95_{experiment}.csv'), index_col=0).squeeze("columns")
    y_ldr_probblocked = \
        pd.read_csv(Path(input_path, f'y_ldr_probblocked_{experiment}.csv'), index_col=0).squeeze("columns")
    y_ldr_condmeantimeblocked = \
        pd.read_csv(Path(input_path, f'y_ldr_condmeantimeblocked_{experiment}.csv'), index_col=0).squeeze("columns")

    # Fit models

    # Queueing models
    ldr_occmean_q_effload_results = \
        crossval_summarize_mm('ldr_occmean_q_effload', 'ldr', 'occmean', X_ldr_q, y_ldr_occmean, scale=False,
                              flavor='load', col_idx_arate=0, col_idx_meansvctime=21)

    ldr_occp95_q_sqrteffload_results = \
        crossval_summarize_mm('ldr_occp95_q_sqrteffload', 'ldr', 'occp95', X_ldr_q, y_ldr_occp95, scale=False,
                              flavor='sqrtload', col_idx_arate=0, col_idx_meansvctime=21, load_pctile=0.95)

    probblocked_q_erlangc_results = \
        crossval_summarize_mm('ldr_probblocked_q_erlangc', 'ldr', 'probblocked',
                              X_ldr_q, y_ldr_probblocked,
                              scale=False, fit_intercept=True,
                              flavor='erlangc', col_idx_arate=0, col_idx_meansvctime=5, col_idx_numservers=7)

    condmeantimeblocked_q_mgc_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_q_mgc', 'ldr', 'condmeantimeblocked',
                              X_ldr_q, y_ldr_condmeantimeblocked,
                              scale=False, fit_intercept=True,
                              flavor='condmeanwaitldr', col_idx_arate=0, col_idx_meansvctime=5, col_idx_numservers=7,
                              col_idx_cv2svctime=18)

    # Linear models using only queueing approximation terms
    ldr_occmean_onlyq_lm_results = \
        crossval_summarize_mm('ldr_occmean_onlyq_lm', 'ldr', 'occmean',
                              X_ldr_occmean_onlyq, y_ldr_occmean, scale=False, flavor='lm')

    ldr_occp95_onlyq_lm_results = \
        crossval_summarize_mm('ldr_occp95_onlyq_lm', 'ldr', 'occp95',
                              X_ldr_occp95_onlyq, y_ldr_occp95, scale=False, flavor='lm')

    probblocked_onlyq_lm_results = \
        crossval_summarize_mm('ldr_probblocked_onlyq_lm', 'ldr', 'probblocked',
                              X_ldr_probblocked_onlyq, y_ldr_probblocked, scale=False, flavor='lm')

    condmeantimeblocked_onlyq_lm_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_onlyq_lm', 'ldr', 'condmeantimeblocked',
                              X_ldr_condmeantimeblocked_onlyq, y_ldr_condmeantimeblocked, scale=False, flavor='lm')

    # Linear regression (lm)
    ldr_occmean_basicq_lm_results = \
        crossval_summarize_mm('ldr_occmean_basicq_lm', 'ldr', 'occmean',
                              X_ldr_basicq, y_ldr_occmean, scale=False, flavor='lm')

    ldr_occmean_q_lm_results = \
        crossval_summarize_mm('ldr_occmean_q_lm', 'ldr', 'occmean',
                              X_ldr_q, y_ldr_occmean, scale=False, flavor='lm')

    ldr_occmean_noq_lm_results = \
        crossval_summarize_mm('ldr_occmean_noq_lm', 'ldr', 'occmean',
                              X_ldr_noq, y_ldr_occmean, scale=False, flavor='lm')

    ldr_occp95_basicq_lm_results = \
        crossval_summarize_mm('ldr_occp95_basicq_lm', 'ldr', 'occp95',
                              X_ldr_basicq, y_ldr_occp95, scale=False, flavor='lm')

    ldr_occp95_q_lm_results = \
        crossval_summarize_mm('ldr_occp95_q_lm', 'ldr', 'occp95', X_ldr_q, y_ldr_occp95, scale=False, flavor='lm')

    ldr_occp95_noq_lm_results = \
        crossval_summarize_mm('ldr_occp95_noq_lm', 'ldr', 'occp95', X_ldr_noq, y_ldr_occp95, scale=False, flavor='lm')

    probblocked_basicq_lm_results = \
        crossval_summarize_mm('ldr_probblocked_basicq_lm', 'ldr', 'probblocked',
                              X_ldr_basicq, y_ldr_probblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    probblocked_q_lm_results = \
        crossval_summarize_mm('ldr_probblocked_q_lm', 'ldr', 'probblocked',
                              X_ldr_q, y_ldr_probblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    probblocked_noq_lm_results = \
        crossval_summarize_mm('ldr_probblocked_noq_lm', 'ldr', 'probblocked',
                              X_ldr_noq, y_ldr_probblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    condmeantimeblocked_basicq_lm_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_basicq_lm', 'ldr', 'condmeantimeblocked',
                              X_ldr_basicq, y_ldr_condmeantimeblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    condmeantimeblocked_q_lm_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_q_lm', 'ldr', 'condmeantimeblocked',
                              X_ldr_q, y_ldr_condmeantimeblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    condmeantimeblocked_noq_lm_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_noq_lm', 'ldr', 'condmeantimeblocked',
                              X_ldr_noq, y_ldr_condmeantimeblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    # LassoCV (lassocv)
    ldr_occmean_basicq_lassocv_results = \
        crossval_summarize_mm('ldr_occmean_basicq_lassocv', 'ldr', 'occmean', X_ldr_basicq, y_ldr_occmean,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    ldr_occmean_q_lassocv_results = \
        crossval_summarize_mm('ldr_occmean_q_lassocv', 'ldr', 'occmean', X_ldr_q, y_ldr_occmean,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    ldr_occmean_noq_lassocv_results = \
        crossval_summarize_mm('ldr_occmean_noq_lassocv', 'ldr', 'occmean', X_ldr_noq, y_ldr_occmean,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    ldr_occp95_basicq_lassocv_results = \
        crossval_summarize_mm('ldr_occp95_basicq_lassocv', 'ldr', 'occp95', X_ldr_basicq, y_ldr_occp95,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    ldr_occp95_q_lassocv_results = \
        crossval_summarize_mm('ldr_occp95_q_lassocv', 'ldr', 'occp95', X_ldr_q, y_ldr_occp95,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    ldr_occp95_noq_lassocv_results = \
        crossval_summarize_mm('ldr_occp95_noq_lassocv', 'ldr', 'occp95', X_ldr_noq, y_ldr_occp95,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    probblocked_basicq_lassocv_results = \
        crossval_summarize_mm('ldr_probblocked_basicq_lassocv', 'ldr', 'probblocked',
                              X_ldr_basicq, y_ldr_probblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    probblocked_q_lassocv_results = \
        crossval_summarize_mm('ldr_probblocked_q_lassocv', 'ldr', 'probblocked',
                              X_ldr_q, y_ldr_probblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    probblocked_noq_lassocv_results = \
        crossval_summarize_mm('ldr_probblocked_noq_lassocv', 'ldr', 'probblocked',
                              X_ldr_noq, y_ldr_probblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    condmeantimeblocked_basicq_lassocv_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_basicq_lassocv', 'ldr', 'condmeantimeblocked',
                              X_ldr_basicq, y_ldr_condmeantimeblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    condmeantimeblocked_q_lassocv_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_q_lassocv', 'ldr', 'condmeantimeblocked',
                              X_ldr_q, y_ldr_condmeantimeblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    condmeantimeblocked_noq_lassocv_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_noq_lassocv', 'ldr', 'condmeantimeblocked',
                              X_ldr_noq, y_ldr_condmeantimeblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    # Polynomial regression (poly)
    ldr_occmean_basicq_poly_results = \
        crossval_summarize_mm('ldr_occmean_basicq_poly', 'ldr', 'occmean',
                              X_ldr_basicq, y_ldr_occmean, scale=False, flavor='poly')

    ldr_occmean_q_poly_results = \
        crossval_summarize_mm('ldr_occmean_q_poly', 'ldr', 'occmean',
                              X_ldr_q, y_ldr_occmean, scale=False, flavor='poly')

    ldr_occmean_noq_poly_results = \
        crossval_summarize_mm('ldr_occmean_noq_poly', 'ldr', 'occmean',
                              X_ldr_noq, y_ldr_occmean, scale=False,
                              flavor='poly')

    ldr_occp95_basicq_poly_results = \
        crossval_summarize_mm('ldr_occp95_basicq_poly', 'ldr', 'occp95',
                              X_ldr_basicq, y_ldr_occp95, scale=False, flavor='poly')

    ldr_occp95_q_poly_results = \
        crossval_summarize_mm('ldr_occp95_q_poly', 'ldr', 'occp95',
                              X_ldr_q, y_ldr_occp95, scale=False, flavor='poly')

    ldr_occp95_noq_poly_results = \
        crossval_summarize_mm('ldr_occp95_noq_poly', 'ldr', 'occp95',
                              X_ldr_noq, y_ldr_occp95, scale=False, flavor='poly')

    probblocked_basicq_poly_results = \
        crossval_summarize_mm('ldr_probblocked_basicq_poly', 'ldr', 'probblocked',
                              X_ldr_basicq, y_ldr_probblocked,
                              scale=False, flavor='poly')

    probblocked_q_poly_results = \
        crossval_summarize_mm('ldr_probblocked_q_poly', 'ldr', 'probblocked',
                              X_ldr_q, y_ldr_probblocked,
                              scale=False, flavor='poly')

    probblocked_noq_poly_results = \
        crossval_summarize_mm('ldr_probblocked_noq_poly', 'ldr', 'probblocked',
                              X_ldr_noq, y_ldr_probblocked,
                              scale=False, flavor='poly')

    condmeantimeblocked_basicq_poly_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_basicq_poly', 'ldr', 'condmeantimeblocked',
                              X_ldr_basicq, y_ldr_condmeantimeblocked,
                              scale=False, flavor='poly')

    condmeantimeblocked_q_poly_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_q_poly', 'ldr', 'condmeantimeblocked',
                              X_ldr_q, y_ldr_condmeantimeblocked,
                              scale=False, flavor='poly')

    condmeantimeblocked_noq_poly_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_noq_poly', 'ldr', 'condmeantimeblocked',
                              X_ldr_noq, y_ldr_condmeantimeblocked,
                              scale=False, flavor='poly')

    # Random forest (rf)
    ldr_occmean_basicq_rf_results = \
        crossval_summarize_mm('ldr_occmean_basicq_rf', 'ldr', 'occmean',
                              X_ldr_basicq, y_ldr_occmean, scale=False, flavor='rf')

    ldr_occmean_q_rf_results = \
        crossval_summarize_mm('ldr_occmean_q_rf', 'ldr', 'occmean',
                              X_ldr_q, y_ldr_occmean, scale=False, flavor='rf')

    ldr_occmean_noq_rf_results = \
        crossval_summarize_mm('ldr_occmean_noq_rf', 'ldr', 'occmean',
                              X_ldr_noq, y_ldr_occmean, scale=False, flavor='rf')

    ldr_occp95_basicq_rf_results = \
        crossval_summarize_mm('ldr_occp95_basicq_rf', 'ldr', 'occp95',
                              X_ldr_basicq, y_ldr_occp95, scale=False, flavor='rf')

    ldr_occp95_q_rf_results = \
        crossval_summarize_mm('ldr_occp95_q_rf', 'ldr', 'occp95',
                              X_ldr_q, y_ldr_occp95, scale=False, flavor='rf')

    ldr_occp95_noq_rf_results = \
        crossval_summarize_mm('ldr_occp95_noq_rf', 'ldr', 'occp95',
                              X_ldr_noq, y_ldr_occp95, scale=False, flavor='rf')

    probblocked_basicq_rf_results = \
        crossval_summarize_mm('ldr_probblocked_basicq_rf', 'ldr', 'probblocked',
                              X_ldr_basicq, y_ldr_probblocked,
                              scale=False, flavor='rf')

    probblocked_q_rf_results = \
        crossval_summarize_mm('ldr_probblocked_q_rf', 'ldr', 'probblocked',
                              X_ldr_q, y_ldr_probblocked,
                              scale=False, flavor='rf')

    probblocked_noq_rf_results = \
        crossval_summarize_mm('ldr_probblocked_noq_rf', 'ldr', 'probblocked',
                              X_ldr_noq, y_ldr_probblocked,
                              scale=False, flavor='rf')

    condmeantimeblocked_basicq_rf_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_basicq_rf', 'ldr', 'condmeantimeblocked',
                              X_ldr_basicq, y_ldr_condmeantimeblocked,
                              scale=False, flavor='rf')

    condmeantimeblocked_q_rf_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_q_rf', 'ldr', 'condmeantimeblocked',
                              X_ldr_q, y_ldr_condmeantimeblocked,
                              scale=False, flavor='rf')

    condmeantimeblocked_noq_rf_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_noq_rf', 'ldr', 'condmeantimeblocked',
                              X_ldr_noq, y_ldr_condmeantimeblocked,
                              scale=False, flavor='rf')

    # Support vector regression (svr)
    ldr_occmean_basicq_svr_results = \
        crossval_summarize_mm('ldr_occmean_basicq_svr', 'ldr', 'occmean',
                              X_ldr_basicq, y_ldr_occmean, flavor='svr', scale=True)

    ldr_occmean_q_svr_results = \
        crossval_summarize_mm('ldr_occmean_q_svr', 'ldr', 'occmean',
                              X_ldr_q, y_ldr_occmean, flavor='svr', scale=True)

    ldr_occmean_noq_svr_results = \
        crossval_summarize_mm('ldr_occmean_noq_svr', 'ldr', 'occmean',
                              X_ldr_noq, y_ldr_occmean, flavor='svr', scale=True)

    ldr_occp95_basicq_svr_results = \
        crossval_summarize_mm('ldr_occp95_basicq_svr', 'ldr', 'occp95',
                              X_ldr_basicq, y_ldr_occp95, flavor='svr', scale=True)

    ldr_occp95_q_svr_results = \
        crossval_summarize_mm('ldr_occp95_q_svr', 'ldr', 'occp95',
                              X_ldr_q, y_ldr_occp95, flavor='svr', scale=True)

    ldr_occp95_noq_svr_results = \
        crossval_summarize_mm('ldr_occp95_noq_svr', 'ldr', 'occp95',
                              X_ldr_noq, y_ldr_occp95, flavor='svr', scale=True)

    probblocked_basicq_svr_results = \
        crossval_summarize_mm('ldr_probblocked_basicq_svr', 'ldr', 'probblocked',
                              X_ldr_basicq, y_ldr_probblocked,
                              scale=True, flavor='svr')

    probblocked_q_svr_results = \
        crossval_summarize_mm('ldr_probblocked_q_svr', 'ldr', 'probblocked',
                              X_ldr_q, y_ldr_probblocked,
                              scale=True, flavor='svr')

    probblocked_noq_svr_results = \
        crossval_summarize_mm('ldr_probblocked_noq_svr', 'ldr', 'probblocked',
                              X_ldr_noq, y_ldr_probblocked,
                              scale=True, flavor='svr')

    condmeantimeblocked_basicq_svr_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_basicq_svr', 'ldr', 'condmeantimeblocked',
                              X_ldr_basicq, y_ldr_condmeantimeblocked,
                              scale=True, flavor='svr')

    condmeantimeblocked_q_svr_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_q_svr', 'ldr', 'condmeantimeblocked',
                              X_ldr_q, y_ldr_condmeantimeblocked,
                              scale=True, flavor='svr')

    condmeantimeblocked_noq_svr_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_noq_svr', 'ldr', 'condmeantimeblocked',
                              X_ldr_noq, y_ldr_condmeantimeblocked,
                              scale=True, flavor='svr')

    # MLPRegressor Neural net (nn)
    ldr_occmean_basicq_nn_results = \
        crossval_summarize_mm('ldr_occmean_basicq_nn', 'ldr', 'occmean',
                              X_ldr_basicq, y_ldr_occmean, flavor='nn', scale=True)

    ldr_occmean_q_nn_results = \
        crossval_summarize_mm('ldr_occmean_q_nn', 'ldr', 'occmean',
                              X_ldr_q, y_ldr_occmean, flavor='nn', scale=True)

    ldr_occmean_noq_nn_results = \
        crossval_summarize_mm('ldr_occmean_noq_nn', 'ldr', 'occmean', X_ldr_noq, y_ldr_occmean, flavor='nn', scale=True)

    ldr_occp95_basicq_nn_results = \
        crossval_summarize_mm('ldr_occp95_basicq_nn', 'ldr', 'occp95',
                              X_ldr_basicq, y_ldr_occp95, flavor='nn', scale=True)

    ldr_occp95_q_nn_results = \
        crossval_summarize_mm('ldr_occp95_q_nn', 'ldr', 'occp95',
                              X_ldr_q, y_ldr_occp95, flavor='nn', scale=True)

    ldr_occp95_noq_nn_results = \
        crossval_summarize_mm('ldr_occp95_noq_nn', 'ldr', 'occp95',
                              X_ldr_noq, y_ldr_occp95, flavor='nn', scale=True)

    probblocked_basicq_nn_results = \
        crossval_summarize_mm('ldr_probblocked_basicq_nn', 'ldr', 'probblocked',
                              X_ldr_basicq, y_ldr_probblocked,
                              scale=True, flavor='nn')

    probblocked_q_nn_results = \
        crossval_summarize_mm('ldr_probblocked_q_nn', 'ldr', 'probblocked',
                              X_ldr_q, y_ldr_probblocked,
                              scale=True, flavor='nn')

    probblocked_noq_nn_results = \
        crossval_summarize_mm('ldr_probblocked_noq_nn', 'ldr', 'probblocked',
                              X_ldr_noq, y_ldr_probblocked,
                              scale=True, flavor='nn')

    condmeantimeblocked_basicq_nn_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_basicq_nn', 'ldr', 'condmeantimeblocked',
                              X_ldr_basicq, y_ldr_condmeantimeblocked,
                              scale=True, flavor='nn')

    condmeantimeblocked_q_nn_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_q_nn', 'ldr', 'condmeantimeblocked',
                              X_ldr_q, y_ldr_condmeantimeblocked,
                              scale=True, flavor='nn')

    condmeantimeblocked_noq_nn_results = \
        crossval_summarize_mm('ldr_condmeantimeblocked_noq_nn', 'ldr', 'condmeantimeblocked',
                              X_ldr_noq, y_ldr_condmeantimeblocked,
                              scale=True, flavor='nn')

    ldr_results = {'ldr_occmean_basicq_lm_results': ldr_occmean_basicq_lm_results,
                   'ldr_occmean_q_lm_results': ldr_occmean_q_lm_results,
                   'ldr_occmean_noq_lm_results': ldr_occmean_noq_lm_results,
                   'ldr_occmean_basicq_lassocv_results': ldr_occmean_basicq_lassocv_results,
                   'ldr_occmean_q_lassocv_results': ldr_occmean_q_lassocv_results,
                   'ldr_occmean_noq_lassocv_results': ldr_occmean_noq_lassocv_results,
                   'ldr_occmean_basicq_poly_results': ldr_occmean_basicq_poly_results,
                   'ldr_occmean_q_poly_results': ldr_occmean_q_poly_results,
                   'ldr_occmean_noq_poly_results': ldr_occmean_noq_poly_results,
                   'ldr_occmean_basicq_rf_results': ldr_occmean_basicq_rf_results,
                   'ldr_occmean_q_rf_results': ldr_occmean_q_rf_results,
                   'ldr_occmean_noq_rf_results': ldr_occmean_noq_rf_results,
                   'ldr_occmean_basicq_svr_results': ldr_occmean_basicq_svr_results,
                   'ldr_occmean_q_svr_results': ldr_occmean_q_svr_results,
                   'ldr_occmean_noq_svr_results': ldr_occmean_noq_svr_results,
                   'ldr_occmean_basicq_nn_results': ldr_occmean_basicq_nn_results,
                   'ldr_occmean_q_nn_results': ldr_occmean_q_nn_results,
                   'ldr_occmean_noq_nn_results': ldr_occmean_noq_nn_results,
                   'ldr_occmean_q_effload_results': ldr_occmean_q_effload_results,
                   'ldr_occp95_basicq_lm_results': ldr_occp95_basicq_lm_results,
                   'ldr_occp95_q_lm_results': ldr_occp95_q_lm_results,
                   'ldr_occp95_noq_lm_results': ldr_occp95_noq_lm_results,
                   'ldr_occp95_basicq_lassocv_results': ldr_occp95_basicq_lassocv_results,
                   'ldr_occp95_q_lassocv_results': ldr_occp95_q_lassocv_results,
                   'ldr_occp95_noq_lassocv_results': ldr_occp95_noq_lassocv_results,
                   'ldr_occp95_basicq_poly_results': ldr_occp95_basicq_poly_results,
                   'ldr_occp95_q_poly_results': ldr_occp95_q_poly_results,
                   'ldr_occp95_noq_poly_results': ldr_occp95_noq_poly_results,
                   'ldr_occp95_basicq_rf_results': ldr_occp95_basicq_rf_results,
                   'ldr_occp95_q_rf_results': ldr_occp95_q_rf_results,
                   'ldr_occp95_noq_rf_results': ldr_occp95_noq_rf_results,
                   'ldr_occp95_basicq_svr_results': ldr_occp95_basicq_svr_results,
                   'ldr_occp95_q_svr_results': ldr_occp95_q_svr_results,
                   'ldr_occp95_noq_svr_results': ldr_occp95_noq_svr_results,
                   'ldr_occp95_basicq_nn_results': ldr_occp95_basicq_nn_results,
                   'ldr_occp95_q_nn_results': ldr_occp95_q_nn_results,
                   'ldr_occp95_noq_nn_results': ldr_occp95_noq_nn_results,
                   'ldr_occp95_q_sqrteffload_results': ldr_occp95_q_sqrteffload_results,
                   'ldr_probblocked_basicq_lm_results': probblocked_basicq_lm_results,
                   'ldr_probblocked_q_lm_results': probblocked_q_lm_results,
                   'ldr_probblocked_noq_lm_results': probblocked_noq_lm_results,
                   'ldr_probblocked_basicq_lassocv_results': probblocked_basicq_lassocv_results,
                   'ldr_probblocked_q_lassocv_results': probblocked_q_lassocv_results,
                   'ldr_probblocked_noq_lassocv_results': probblocked_noq_lassocv_results,
                   'ldr_probblocked_basicq_poly_results': probblocked_basicq_poly_results,
                   'ldr_probblocked_q_poly_results': probblocked_q_poly_results,
                   'ldr_probblocked_noq_poly_results': probblocked_noq_poly_results,
                   'ldr_probblocked_basicq_rf_results': probblocked_basicq_rf_results,
                   'ldr_probblocked_q_rf_results': probblocked_q_rf_results,
                   'ldr_probblocked_noq_rf_results': probblocked_noq_rf_results,
                   'ldr_probblocked_basicq_svr_results': probblocked_basicq_svr_results,
                   'ldr_probblocked_q_svr_results': probblocked_q_svr_results,
                   'ldr_probblocked_noq_svr_results': probblocked_noq_svr_results,
                   'ldr_probblocked_basicq_nn_results': probblocked_basicq_nn_results,
                   'ldr_probblocked_q_nn_results': probblocked_q_nn_results,
                   'ldr_probblocked_noq_nn_results': probblocked_noq_nn_results,
                   'ldr_probblocked_q_erlangc_results': probblocked_q_erlangc_results,
                   'ldr_condmeantimeblocked_basicq_lm_results': condmeantimeblocked_basicq_lm_results,
                   'ldr_condmeantimeblocked_q_lm_results': condmeantimeblocked_q_lm_results,
                   'ldr_condmeantimeblocked_noq_lm_results': condmeantimeblocked_noq_lm_results,
                   'ldr_condmeantimeblocked_basicq_lassocv_results': condmeantimeblocked_basicq_lassocv_results,
                   'ldr_condmeantimeblocked_q_lassocv_results': condmeantimeblocked_q_lassocv_results,
                   'ldr_condmeantimeblocked_noq_lassocv_results': condmeantimeblocked_noq_lassocv_results,
                   'ldr_condmeantimeblocked_basicq_poly_results': condmeantimeblocked_basicq_poly_results,
                   'ldr_condmeantimeblocked_q_poly_results': condmeantimeblocked_q_poly_results,
                   'ldr_condmeantimeblocked_noq_poly_results': condmeantimeblocked_noq_poly_results,
                   'ldr_condmeantimeblocked_basicq_rf_results': condmeantimeblocked_basicq_rf_results,
                   'ldr_condmeantimeblocked_q_rf_results': condmeantimeblocked_q_rf_results,
                   'ldr_condmeantimeblocked_noq_rf_results': condmeantimeblocked_noq_rf_results,
                   'ldr_condmeantimeblocked_basicq_svr_results': condmeantimeblocked_basicq_svr_results,
                   'ldr_condmeantimeblocked_q_svr_results': condmeantimeblocked_q_svr_results,
                   'ldr_condmeantimeblocked_noq_svr_results': condmeantimeblocked_noq_svr_results,
                   'ldr_condmeantimeblocked_basicq_nn_results': condmeantimeblocked_basicq_nn_results,
                   'ldr_condmeantimeblocked_q_nn_results': condmeantimeblocked_q_nn_results,
                   'ldr_condmeantimeblocked_noq_nn_results': condmeantimeblocked_noq_nn_results,
                   'ldr_condmeantimeblocked_q_mgc_results': condmeantimeblocked_q_mgc_results,
                   'ldr_occmean_onlyq_lm_results': ldr_occmean_onlyq_lm_results,
                   'ldr_occp95_onlyq_lm_results': ldr_occp95_onlyq_lm_results,
                   'ldr_probblocked_onlyq_lm_results': probblocked_onlyq_lm_results,
                   'ldr_condmeantimeblocked_onlyq_lm_results': condmeantimeblocked_onlyq_lm_results
                   }

    create_cv_plots(experiment, unit, ldr_results, figures_path)
    create_coeff_plots(experiment, unit, ldr_results, figures_path)

    metrics_df = create_metrics_df(ldr_results)
    metrics_df.to_csv(metrics_path_filename, index=False)

    predictions_df = create_predictions_df(ldr_results)
    predictions_df.to_csv(Path(output_path, f"{experiment}_{unit}_predictions.csv"), index=False)

    sys.setrecursionlimit(10000)
    # Pickle the results
    with open(pickle_path_filename, 'wb') as persisted_file:
        pickle.dump(ldr_results, persisted_file)


def process_command_line(argv=None):
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='mm_run_fits_ldr',
                                     description='Fit metamodels for LDR')

    # Add arguments
    parser.add_argument(
        "experiment", type=str,
        help="Experiment ID; used in folder and filenames"
    )

    parser.add_argument(
        "input_path", type=str,
        help="Path containing X and y csv files"
    )

    parser.add_argument(
        "output_path", type=str,
        help="Path for model fit outputs (except plots)"
    )

    parser.add_argument(
        "plots_path", type=str,
        help="Path for output plots"
    )

    # do the parsing
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    """
    Main driver program

    Parameters
    ----------
    argv

    Returns
    -------

    """

    # Parse command line arguments
    args = process_command_line(argv)

    # fit models
    fit_models(args.experiment,
               Path(args.input_path), Path(args.output_path), Path(args.plots_path))

    return 0


if __name__ == '__main__':
    sys.exit(main())
