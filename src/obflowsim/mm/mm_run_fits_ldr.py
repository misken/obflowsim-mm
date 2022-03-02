import sys
import argparse
from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from obflowsim.mm.mm_fitting import crossval_summarize_mm
from obflowsim.mm.mm_process_fitted_models import create_cv_plots, create_coeff_plots, create_metrics_df

# Make sure no interactive plotting happens during plot generation
plt.ioff()

UNIT = "ldr"


def fit_models(experiment):
    """
    Fit metamodels for LDR unit

    Parameters
    ----------
    experiment : str, identifier used in folder and filenames

    Returns
    -------

    """
    unit = UNIT
    input_path = Path(f"input/{experiment}")
    output_path = Path(f"output/{experiment}")
    figures_path = Path(f"output/{experiment}", "figures")

    metrics_path_filename = Path(output_path, f"{experiment}_{unit}_metrics.csv")
    pickle_path_filename = Path(output_path, f"{experiment}_{unit}_results.pkl")

    # X matrices
    X_ldr_noq = pd.read_csv(Path(input_path, f'X_ldr_noq_{experiment}.csv'), index_col=0)
    X_ldr_basicq = pd.read_csv(Path(input_path, f'X_ldr_basicq_{experiment}.csv'), index_col=0)
    X_ldr_q = pd.read_csv(Path(input_path, f'X_ldr_q_{experiment}.csv'), index_col=0)

    X_ldr_occmean_onlyq = pd.read_csv(Path(input_path, f'X_ldr_occmean_onlyq_{experiment}.csv'), index_col=0)
    X_ldr_occp95_onlyq = pd.read_csv(Path(input_path, f'X_ldr_occp95_onlyq_{experiment}.csv'), index_col=0)
    X_ldr_probblockedbypp_onlyq = pd.read_csv(Path(input_path,
                                                   f'X_ldr_probblockedbypp_onlyq_{experiment}.csv'),
                                              index_col=0)
    X_ldr_condmeantimeblockedbypp_onlyq = \
        pd.read_csv(Path(input_path, f'X_ldr_condmeantimeblockedbypp_onlyq_{experiment}.csv'), index_col=0)

    # y vectors
    y_ldr_occmean = pd.read_csv(Path(input_path, f'y_ldr_occmean_{experiment}.csv'), index_col=0, squeeze=True)
    y_ldr_occp95 = pd.read_csv(Path(input_path, f'y_ldr_occp95_{experiment}.csv'), index_col=0, squeeze=True)
    y_ldr_probblocked = \
        pd.read_csv(Path(input_path, f'y_ldr_probblocked_{experiment}.csv'), index_col=0, squeeze=True)
    y_ldr_condmeantimeblocked = \
        pd.read_csv(Path(input_path, f'y_ldr_condmeantimeblocked_{experiment}.csv'), index_col=0, squeeze=True)

    # Fit models

    # Queueing models
    ldr_occmean_q_load_results = \
        crossval_summarize_mm('ldr_occmean_q_effload', 'ldr', 'occmean', X_ldr_q, y_ldr_occmean, scale=False,
                              flavor='load', col_idx_arate=0, col_idx_meansvctime=21)

    ldr_occp95_q_sqrtload_results = \
        crossval_summarize_mm('ldr_occp95_q_sqrteffload', 'ldr', 'occp95', X_ldr_q, y_ldr_occp95, scale=False,
                              flavor='sqrtload', col_idx_arate=0, col_idx_meansvctime=21, load_pctile=0.95)

    probblockedbypp_q_erlangc_results = \
        crossval_summarize_mm('ldr_probblockedbypp_q_erlangc', 'ldr', 'probblockedbypp',
                              X_ldr_q, y_ldr_probblocked,
                              scale=False, fit_intercept=True,
                              flavor='erlangc', col_idx_arate=0, col_idx_meansvctime=5, col_idx_numservers=7)

    condmeantimeblockedbypp_q_mgc_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_q_mgc', 'ldr', 'condmeantimeblockedbypp',
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

    probblockedbypp_onlyq_lm_results = \
        crossval_summarize_mm('ldr_probblockedbypp_onlyq_lm', 'ldr', 'probblockedbypp',
                              X_ldr_probblockedbypp_onlyq, y_ldr_probblocked, scale=False, flavor='lm')

    condmeantimeblockedbypp_onlyq_lm_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_onlyq_lm', 'ldr', 'condmeantimeblockedbypp',
                              X_ldr_condmeantimeblockedbypp_onlyq, y_ldr_condmeantimeblocked, scale=False, flavor='lm')

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

    probblockedbypp_basicq_lm_results = \
        crossval_summarize_mm('ldr_probblockedbypp_basicq_lm', 'ldr', 'probblockedbypp',
                              X_ldr_basicq, y_ldr_probblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    probblockedbypp_q_lm_results = \
        crossval_summarize_mm('ldr_probblockedbypp_q_lm', 'ldr', 'probblockedbypp',
                              X_ldr_q, y_ldr_probblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    probblockedbypp_noq_lm_results = \
        crossval_summarize_mm('ldr_probblockedbypp_noq_lm', 'ldr', 'probblockedbypp',
                              X_ldr_noq, y_ldr_probblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    condmeantimeblockedbypp_basicq_lm_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_basicq_lm', 'ldr', 'condmeantimeblockedbypp',
                              X_ldr_basicq, y_ldr_condmeantimeblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    condmeantimeblockedbypp_q_lm_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_q_lm', 'ldr', 'condmeantimeblockedbypp',
                              X_ldr_q, y_ldr_condmeantimeblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    condmeantimeblockedbypp_noq_lm_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_noq_lm', 'ldr', 'condmeantimeblockedbypp',
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

    probblockedbypp_basicq_lassocv_results = \
        crossval_summarize_mm('ldr_probblockedbypp_basicq_lassocv', 'ldr', 'probblockedbypp',
                              X_ldr_basicq, y_ldr_probblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    probblockedbypp_q_lassocv_results = \
        crossval_summarize_mm('ldr_probblockedbypp_q_lassocv', 'ldr', 'probblockedbypp',
                              X_ldr_q, y_ldr_probblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    probblockedbypp_noq_lassocv_results = \
        crossval_summarize_mm('ldr_probblockedbypp_noq_lassocv', 'ldr', 'probblockedbypp',
                              X_ldr_noq, y_ldr_probblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    condmeantimeblockedbypp_basicq_lassocv_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_basicq_lassocv', 'ldr', 'condmeantimeblockedbypp',
                              X_ldr_basicq, y_ldr_condmeantimeblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    condmeantimeblockedbypp_q_lassocv_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_q_lassocv', 'ldr', 'condmeantimeblockedbypp',
                              X_ldr_q, y_ldr_condmeantimeblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    condmeantimeblockedbypp_noq_lassocv_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_noq_lassocv', 'ldr', 'condmeantimeblockedbypp',
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

    probblockedbypp_basicq_poly_results = \
        crossval_summarize_mm('ldr_probblockedbypp_basicq_poly', 'ldr', 'probblockedbypp',
                              X_ldr_basicq, y_ldr_probblocked,
                              scale=False, flavor='poly')

    probblockedbypp_q_poly_results = \
        crossval_summarize_mm('ldr_probblockedbypp_q_poly', 'ldr', 'probblockedbypp',
                              X_ldr_q, y_ldr_probblocked,
                              scale=False, flavor='poly')

    probblockedbypp_noq_poly_results = \
        crossval_summarize_mm('ldr_probblockedbypp_noq_poly', 'ldr', 'probblockedbypp',
                              X_ldr_noq, y_ldr_probblocked,
                              scale=False, flavor='poly')

    condmeantimeblockedbypp_basicq_poly_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_basicq_poly', 'ldr', 'condmeantimeblockedbypp',
                              X_ldr_basicq, y_ldr_condmeantimeblocked,
                              scale=False, flavor='poly')

    condmeantimeblockedbypp_q_poly_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_q_poly', 'ldr', 'condmeantimeblockedbypp',
                              X_ldr_q, y_ldr_condmeantimeblocked,
                              scale=False, flavor='poly')

    condmeantimeblockedbypp_noq_poly_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_noq_poly', 'ldr', 'condmeantimeblockedbypp',
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

    probblockedbypp_basicq_rf_results = \
        crossval_summarize_mm('ldr_probblockedbypp_q_rf', 'ldr', 'probblockedbypp',
                              X_ldr_basicq, y_ldr_probblocked,
                              scale=False, flavor='rf')

    probblockedbypp_q_rf_results = \
        crossval_summarize_mm('ldr_probblockedbypp_q_rf', 'ldr', 'probblockedbypp',
                              X_ldr_q, y_ldr_probblocked,
                              scale=False, flavor='rf')

    probblockedbypp_noq_rf_results = \
        crossval_summarize_mm('ldr_probblockedbypp_noq_rf', 'ldr', 'probblockedbypp',
                              X_ldr_noq, y_ldr_probblocked,
                              scale=False, flavor='rf')

    condmeantimeblockedbypp_basicq_rf_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_basicq_rf', 'ldr', 'condmeantimeblockedbypp',
                              X_ldr_basicq, y_ldr_condmeantimeblocked,
                              scale=False, flavor='rf')

    condmeantimeblockedbypp_q_rf_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_q_rf', 'ldr', 'condmeantimeblockedbypp',
                              X_ldr_q, y_ldr_condmeantimeblocked,
                              scale=False, flavor='rf')

    condmeantimeblockedbypp_noq_rf_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_noq_rf', 'ldr', 'condmeantimeblockedbypp',
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

    probblockedbypp_basicq_svr_results = \
        crossval_summarize_mm('ldr_probblockedbypp_basicq_svr', 'ldr', 'probblockedbypp',
                              X_ldr_basicq, y_ldr_probblocked,
                              scale=True, flavor='svr')

    probblockedbypp_q_svr_results = \
        crossval_summarize_mm('ldr_probblockedbypp_q_svr', 'ldr', 'probblockedbypp',
                              X_ldr_q, y_ldr_probblocked,
                              scale=True, flavor='svr')

    probblockedbypp_noq_svr_results = \
        crossval_summarize_mm('ldr_probblockedbypp_noq_svr', 'ldr', 'probblockedbypp',
                              X_ldr_noq, y_ldr_probblocked,
                              scale=True, flavor='svr')

    condmeantimeblockedbypp_basicq_svr_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_basicq_svr', 'ldr', 'condmeantimeblockedbypp',
                              X_ldr_basicq, y_ldr_condmeantimeblocked,
                              scale=True, flavor='svr')

    condmeantimeblockedbypp_q_svr_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_q_svr', 'ldr', 'condmeantime_blockedbyldr',
                              X_ldr_q, y_ldr_condmeantimeblocked,
                              scale=True, flavor='svr')

    condmeantimeblockedbypp_noq_svr_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_noq_svr', 'ldr', 'condmeantimeblockedbypp',
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

    probblockedbypp_basicq_nn_results = \
        crossval_summarize_mm('ldr_probblockedbypp_basicq_nn', 'ldr', 'probblockedbypp',
                              X_ldr_basicq, y_ldr_probblocked,
                              scale=True, flavor='nn')

    probblockedbypp_q_nn_results = \
        crossval_summarize_mm('ldr_probblockedbypp_q_nn', 'ldr', 'probblockedbypp',
                              X_ldr_q, y_ldr_probblocked,
                              scale=True, flavor='nn')

    probblockedbypp_noq_nn_results = \
        crossval_summarize_mm('ldr_probblockedbypp_noq_nn', 'ldr', 'probblockedbypp',
                              X_ldr_noq, y_ldr_probblocked,
                              scale=True, flavor='nn')

    condmeantimeblockedbypp_basicq_nn_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_basicq_nn', 'ldr', 'condmeantimeblockedbypp',
                              X_ldr_basicq, y_ldr_condmeantimeblocked,
                              scale=True, flavor='nn')

    condmeantimeblockedbypp_q_nn_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_q_nn', 'ldr', 'condmeantimeblockedbypp',
                              X_ldr_q, y_ldr_condmeantimeblocked,
                              scale=True, flavor='nn')

    condmeantimeblockedbypp_noq_nn_results = \
        crossval_summarize_mm('ldr_condmeantimeblockedbypp_noq_nn', 'ldr', 'condmeantimeblockedbypp',
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
                   'ldr_occmean_q_load_results': ldr_occmean_q_load_results,
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
                   'ldr_occp95_q_sqrtload_results': ldr_occp95_q_sqrtload_results,
                   'ldr_probblockedbypp_basicq_lm_results': probblockedbypp_basicq_lm_results,
                   'ldr_probblockedbypp_q_lm_results': probblockedbypp_q_lm_results,
                   'ldr_probblockedbypp_noq_lm_results': probblockedbypp_noq_lm_results,
                   'ldr_probblockedbypp_basicq_lassocv_results': probblockedbypp_basicq_lassocv_results,
                   'ldr_probblockedbypp_q_lassocv_results': probblockedbypp_q_lassocv_results,
                   'ldr_probblockedbypp_noq_lassocv_results': probblockedbypp_noq_lassocv_results,
                   'ldr_probblockedbypp_basicq_poly_results': probblockedbypp_basicq_poly_results,
                   'ldr_probblockedbypp_q_poly_results': probblockedbypp_q_poly_results,
                   'ldr_probblockedbypp_noq_poly_results': probblockedbypp_noq_poly_results,
                   'ldr_probblockedbypp_basicq_rf_results': probblockedbypp_basicq_rf_results,
                   'ldr_probblockedbypp_q_rf_results': probblockedbypp_q_rf_results,
                   'ldr_probblockedbypp_noq_rf_results': probblockedbypp_noq_rf_results,
                   'ldr_probblockedbypp_basicq_svr_results': probblockedbypp_basicq_svr_results,
                   'ldr_probblockedbypp_q_svr_results': probblockedbypp_q_svr_results,
                   'ldr_probblockedbypp_noq_svr_results': probblockedbypp_noq_svr_results,
                   'ldr_probblockedbypp_basicq_nn_results': probblockedbypp_basicq_nn_results,
                   'ldr_probblockedbypp_q_nn_results': probblockedbypp_q_nn_results,
                   'ldr_probblockedbypp_noq_nn_results': probblockedbypp_noq_nn_results,
                   'ldr_probblockedbypp_q_erlangc_results': probblockedbypp_q_erlangc_results,
                   'ldr_condmeantimeblockedbypp_basicq_lm_results': condmeantimeblockedbypp_basicq_lm_results,
                   'ldr_condmeantimeblockedbypp_q_lm_results': condmeantimeblockedbypp_q_lm_results,
                   'ldr_condmeantimeblockedbypp_noq_lm_results': condmeantimeblockedbypp_noq_lm_results,
                   'ldr_condmeantimeblockedbypp_basicq_lassocv_results': condmeantimeblockedbypp_basicq_lassocv_results,
                   'ldr_condmeantimeblockedbypp_q_lassocv_results': condmeantimeblockedbypp_q_lassocv_results,
                   'ldr_condmeantimeblockedbypp_noq_lassocv_results': condmeantimeblockedbypp_noq_lassocv_results,
                   'ldr_condmeantimeblockedbypp_basicq_poly_results': condmeantimeblockedbypp_basicq_poly_results,
                   'ldr_condmeantimeblockedbypp_q_poly_results': condmeantimeblockedbypp_q_poly_results,
                   'ldr_condmeantimeblockedbypp_noq_poly_results': condmeantimeblockedbypp_noq_poly_results,
                   'ldr_condmeantimeblockedbypp_basicq_rf_results': condmeantimeblockedbypp_basicq_rf_results,
                   'ldr_condmeantimeblockedbypp_q_rf_results': condmeantimeblockedbypp_q_rf_results,
                   'ldr_condmeantimeblockedbypp_noq_rf_results': condmeantimeblockedbypp_noq_rf_results,
                   'ldr_condmeantimeblockedbypp_basicq_svr_results': condmeantimeblockedbypp_basicq_svr_results,
                   'ldr_condmeantimeblockedbypp_q_svr_results': condmeantimeblockedbypp_q_svr_results,
                   'ldr_condmeantimeblockedbypp_noq_svr_results': condmeantimeblockedbypp_noq_svr_results,
                   'ldr_condmeantimeblockedbypp_basicq_nn_results': condmeantimeblockedbypp_basicq_nn_results,
                   'ldr_condmeantimeblockedbypp_q_nn_results': condmeantimeblockedbypp_q_nn_results,
                   'ldr_condmeantimeblockedbypp_noq_nn_results': condmeantimeblockedbypp_noq_nn_results,
                   'ldr_condmeantimeblockedbypp_q_mgc_results': condmeantimeblockedbypp_q_mgc_results,
                   'ldr_occmean_onlyq_lm_results': ldr_occmean_onlyq_lm_results,
                   'ldr_occp95_onlyq_lm_results': ldr_occp95_onlyq_lm_results,
                   'ldr_probblockedbypp_onlyq_lm_results': probblockedbypp_onlyq_lm_results,
                   'ldr_condmeantimeblockedbypp_onlyq_lm_results': condmeantimeblockedbypp_onlyq_lm_results
                   }

    create_cv_plots(experiment, unit, ldr_results, figures_path)
    create_coeff_plots(experiment, unit, ldr_results, figures_path)

    metrics_df = create_metrics_df(ldr_results)
    metrics_df.to_csv(metrics_path_filename, index=False)

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
    fit_models(args.experiment)

    return 0


if __name__ == '__main__':
    sys.exit(main())
