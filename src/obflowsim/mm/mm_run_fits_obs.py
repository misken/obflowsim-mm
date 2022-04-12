import sys
import argparse
from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from obflowsim.mm.mm_fitting import crossval_summarize_mm
from obflowsim.mm.mm_process_fitted_models import create_cv_plots, create_coeff_plots
from obflowsim.mm.mm_process_fitted_models import create_metrics_df, create_predictions_df

UNIT = "obs"


def fit_models(experiment, input_path, output_path, figures_path):
    """
    Fit metamodels for OBS unit

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
    X_obs_noq = pd.read_csv(Path(input_path, f'X_obs_noq_{experiment}.csv'), index_col=0)
    X_obs_basicq = pd.read_csv(Path(input_path, f'X_obs_basicq_{experiment}.csv'), index_col=0)
    X_obs_q = pd.read_csv(Path(input_path, f'X_obs_q_{experiment}.csv'), index_col=0)

    X_obs_occmean_onlyq = pd.read_csv(Path(input_path, f'X_obs_occmean_onlyq_{experiment}.csv'), index_col=0)
    X_obs_occp95_onlyq = pd.read_csv(Path(input_path, f'X_obs_occp95_onlyq_{experiment}.csv'), index_col=0)
    X_obs_probblocked_onlyq = pd.read_csv(Path(input_path, f'X_obs_probblocked_onlyq_{experiment}.csv'), index_col=0)
    X_obs_condmeantimeblocked_onlyq = \
        pd.read_csv(Path(input_path, f'X_obs_condmeantimeblocked_onlyq_{experiment}.csv'), index_col=0)

    # y vectors
    y_obs_occmean = pd.read_csv(Path(input_path, f'y_obs_occmean_{experiment}.csv'), index_col=0).squeeze("columns")
    y_obs_occp95 = pd.read_csv(Path(input_path, f'y_obs_occp95_{experiment}.csv'), index_col=0).squeeze("columns")
    y_obs_probblocked = pd.read_csv(Path(input_path,
                                                 f'y_obs_probblocked_{experiment}.csv'), index_col=0).squeeze("columns")
    y_obs_condmeantimeblocked = pd.read_csv(Path(input_path,
                                                   f'y_obs_condmeantimeblocked_{experiment}.csv'),
                                              index_col=0).squeeze("columns")

    # Queueing models

    obs_occmean_q_load_results = \
        crossval_summarize_mm('obs_occmean_q_load', 'obs', 'occmean', X_obs_q, y_obs_occmean, scale=False,
                              flavor='load', col_idx_arate=0, col_idx_meansvctime=1)

    obs_occp95_q_sqrtload_results = \
        crossval_summarize_mm('obs_occp95_q_sqrtload', 'obs', 'occp95', X_obs_q, y_obs_occp95, scale=False,
                              flavor='sqrtload', col_idx_arate=0, col_idx_meansvctime=1, load_pctile=0.95)

# TODO The following need eff mean svc time in OBS
    # obs_occmean_q_effload_results = \
    #     crossval_summarize_mm('obs_occmean_q_effload', 'obs', 'occmean', X_obs_q, y_obs_occmean, scale=False,
    #                           flavor='load', col_idx_arate=0, col_idx_meansvctime=19)
    #
    # obs_occp95_q_sqrteffload_results = \
    #     crossval_summarize_mm('obs_occp95_q_sqrteffload', 'obs', 'occp95', X_obs_q, y_obs_occp95, scale=False,
    #                           flavor='sqrtload', col_idx_arate=0, col_idx_meansvctime=19, load_pctile=0.95)

    probblocked_q_erlangc_results = \
        crossval_summarize_mm('obs_probblocked_q_erlangc', 'obs', 'probblocked',
                              X_obs_q, y_obs_probblocked,
                              scale=False, fit_intercept=True,
                              flavor='erlangc', col_idx_arate=0, col_idx_meansvctime=21, col_idx_numservers=4)

    condmeantimeblocked_q_mgc_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_q_mgc', 'obs', 'condmeantimeblocked',
                              X_obs_q, y_obs_condmeantimeblocked,
                              scale=False, fit_intercept=True,
                              flavor='condmeanwaitldr', col_idx_arate=0, col_idx_meansvctime=21, col_idx_numservers=4,
                              col_idx_cv2svctime=18)

    # Linear models using only queueing approximation terms
    obs_occmean_onlyq_lm_results = \
        crossval_summarize_mm('obs_occmean_onlyq_lm', 'obs', 'occmean',
                              X_obs_occmean_onlyq, y_obs_occmean, scale=False, flavor='lm')

    obs_occp95_onlyq_lm_results = \
        crossval_summarize_mm('obs_occp95_onlyq_lm', 'obs', 'occp95',
                              X_obs_occp95_onlyq, y_obs_occp95, scale=False, flavor='lm')

    probblocked_onlyq_lm_results = \
        crossval_summarize_mm('obs_probblocked_onlyq_lm', 'obs', 'probblocked',
                              X_obs_probblocked_onlyq, y_obs_probblocked, scale=False, flavor='lm')

    condmeantimeblocked_onlyq_lm_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_onlyq_lm', 'obs', 'condmeantimeblocked',
                              X_obs_condmeantimeblocked_onlyq, y_obs_condmeantimeblocked, scale=False, flavor='lm')

    ## Linear regression (lm)
    obs_occmean_basicq_lm_results = \
        crossval_summarize_mm('obs_occmean_basicq_lm', 'obs', 'occmean',
                              X_obs_basicq, y_obs_occmean, scale=False, flavor='lm')

    obs_occmean_q_lm_results = \
        crossval_summarize_mm('obs_occmean_q_lm', 'obs', 'occmean',
                              X_obs_q, y_obs_occmean, scale=False, flavor='lm')

    obs_occmean_noq_lm_results = \
        crossval_summarize_mm('obs_occmean_noq_lm', 'obs', 'occmean',
                              X_obs_noq, y_obs_occmean, scale=False, flavor='lm')


    obs_occp95_basicq_lm_results = \
        crossval_summarize_mm('obs_occp95_basicq_lm', 'obs', 'occp95',
                              X_obs_basicq, y_obs_occp95, scale=False, flavor='lm')

    obs_occp95_q_lm_results = \
        crossval_summarize_mm('obs_occp95_q_lm', 'obs', 'occp95',
                              X_obs_q, y_obs_occp95, scale=False, flavor='lm')

    obs_occp95_noq_lm_results = \
        crossval_summarize_mm('obs_occp95_noq_lm', 'obs', 'occp95',
                              X_obs_noq, y_obs_occp95, scale=False, flavor='lm')



    # LassoCV (lassocv)
    obs_occmean_basicq_lassocv_results = \
        crossval_summarize_mm('obs_occmean_basicq_lassocv', 'obs', 'occmean',
                              X_obs_basicq, y_obs_occmean,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    obs_occmean_q_lassocv_results = \
        crossval_summarize_mm('obs_occmean_q_lassocv', 'obs', 'occmean',
                              X_obs_q, y_obs_occmean,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    obs_occmean_noq_lassocv_results = \
        crossval_summarize_mm('obs_occmean_noq_lassocv', 'obs', 'occmean',
                              X_obs_noq, y_obs_occmean,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    obs_occp95_basicq_lassocv_results = \
        crossval_summarize_mm('obs_occp95_basicq_lassocv', 'obs', 'occp95',
                              X_obs_basicq, y_obs_occp95,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    obs_occp95_q_lassocv_results = \
        crossval_summarize_mm('obs_occp95_q_lassocv', 'obs', 'occp95',
                              X_obs_q, y_obs_occp95,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    obs_occp95_noq_lassocv_results = \
        crossval_summarize_mm('obs_occp95_noq_lassocv', 'obs', 'occp95',
                              X_obs_noq, y_obs_occp95,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)




    # Polynomial regression (poly)
    obs_occmean_basicq_poly_results = \
        crossval_summarize_mm('obs_occmean_basicq_poly', 'obs', 'occmean',
                              X_obs_basicq, y_obs_occmean, scale=False, flavor='poly')

    obs_occmean_q_poly_results = \
        crossval_summarize_mm('obs_occmean_q_poly', 'obs', 'occmean',
                              X_obs_q, y_obs_occmean, scale=False, flavor='poly')

    obs_occmean_noq_poly_results = \
        crossval_summarize_mm('obs_occmean_noq_poly', 'obs', 'occmean',
                              X_obs_noq, y_obs_occmean, scale=False, flavor='poly')


    obs_occp95_basicq_poly_results = \
        crossval_summarize_mm('obs_occp95_basicq_poly', 'obs', 'occp95',
                              X_obs_basicq, y_obs_occp95, scale=False, flavor='poly')

    obs_occp95_q_poly_results = \
        crossval_summarize_mm('obs_occp95_q_poly', 'obs', 'occp95',
                              X_obs_q, y_obs_occp95, scale=False, flavor='poly')

    obs_occp95_noq_poly_results = \
        crossval_summarize_mm('obs_occp95_noq_poly', 'obs', 'occp95',
                              X_obs_noq, y_obs_occp95, scale=False, flavor='poly')



    # Random forest (rf)
    obs_occmean_basicq_rf_results = \
        crossval_summarize_mm('obs_occmean_basicq_rf', 'obs', 'occmean',
                              X_obs_basicq, y_obs_occmean, scale=False, flavor='rf')

    obs_occmean_q_rf_results = \
        crossval_summarize_mm('obs_occmean_q_rf', 'obs', 'occmean',
                              X_obs_q, y_obs_occmean, scale=False, flavor='rf')

    obs_occmean_noq_rf_results = \
        crossval_summarize_mm('obs_occmean_noq_rf', 'obs', 'occmean',
                              X_obs_noq, y_obs_occmean, scale=False, flavor='rf')

    obs_occp95_basicq_rf_results = \
        crossval_summarize_mm('obs_occp95_basicq_rf', 'obs', 'occp95',
                              X_obs_basicq, y_obs_occp95, scale=False, flavor='rf')

    obs_occp95_q_rf_results = \
        crossval_summarize_mm('obs_occp95_q_rf', 'obs', 'occp95',
                              X_obs_q, y_obs_occp95, scale=False, flavor='rf')

    obs_occp95_noq_rf_results = \
        crossval_summarize_mm('obs_occp95_noq_rf', 'obs', 'occp95',
                              X_obs_noq, y_obs_occp95, scale=False, flavor='rf')



    # Support vector regression (svr)
    obs_occmean_basicq_svr_results = \
        crossval_summarize_mm('obs_occmean_basicq_svr', 'obs', 'occmean',
                              X_obs_basicq, y_obs_occmean, flavor='svr', scale=True)

    obs_occmean_q_svr_results = \
        crossval_summarize_mm('obs_occmean_q_svr', 'obs', 'occmean',
                              X_obs_q, y_obs_occmean, flavor='svr', scale=True)

    obs_occmean_noq_svr_results = \
        crossval_summarize_mm('obs_occmean_noq_svr', 'obs', 'occmean',
                              X_obs_noq, y_obs_occmean, flavor='svr', scale=True)


    obs_occp95_basicq_svr_results = \
        crossval_summarize_mm('obs_occp95_basicq_svr', 'obs', 'occp95',
                              X_obs_basicq, y_obs_occp95, flavor='svr', scale=True)

    obs_occp95_q_svr_results = \
        crossval_summarize_mm('obs_occp95_q_svr', 'obs', 'occp95',
                              X_obs_q, y_obs_occp95, flavor='svr', scale=True)

    obs_occp95_noq_svr_results = \
        crossval_summarize_mm('obs_occp95_noq_svr', 'obs', 'occp95',
                              X_obs_noq, y_obs_occp95, flavor='svr', scale=True)



    # MLPRegressor Neural net (nn)
    obs_occmean_basicq_nn_results = \
        crossval_summarize_mm('obs_occmean_basicq_nn', 'obs', 'occmean',
                              X_obs_basicq, y_obs_occmean, flavor='nn', scale=True)

    obs_occmean_q_nn_results = \
        crossval_summarize_mm('obs_occmean_q_nn', 'obs', 'occmean',
                              X_obs_q, y_obs_occmean, flavor='nn', scale=True)

    obs_occmean_noq_nn_results = \
        crossval_summarize_mm('obs_occmean_noq_nn', 'obs', 'occmean',
                              X_obs_noq, y_obs_occmean, flavor='nn', scale=True)

    obs_occp95_basicq_nn_results = \
        crossval_summarize_mm('obs_occp95_basicq_nn', 'obs', 'occp95',
                              X_obs_basicq, y_obs_occp95, flavor='nn', scale=True)

    obs_occp95_q_nn_results = \
        crossval_summarize_mm('obs_occp95_q_nn', 'obs', 'occp95',
                              X_obs_q, y_obs_occp95, flavor='nn', scale=True)

    obs_occp95_noq_nn_results = \
        crossval_summarize_mm('obs_occp95_noq_nn', 'obs', 'occp95',
                              X_obs_noq, y_obs_occp95, flavor='nn', scale=True)


    ## Linear regression (lm)


    probblocked_basicq_lm_results = \
        crossval_summarize_mm('obs_probblocked_basicq_lm', 'obs', 'probblocked',
                              X_obs_basicq, y_obs_probblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    probblocked_q_lm_results = \
        crossval_summarize_mm('obs_probblocked_q_lm', 'obs', 'probblocked',
                              X_obs_q, y_obs_probblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    probblocked_noq_lm_results = \
        crossval_summarize_mm('obs_probblocked_noq_lm', 'obs', 'probblocked',
                              X_obs_noq, y_obs_probblocked,
                              scale=False, fit_intercept=True, flavor='lm')

    condmeantimeblocked_basicq_lm_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_basicq_lm', 'obs', 'condmeantimeblocked',
                              X_obs_basicq, y_obs_condmeantimeblocked,
                                                   scale=False, fit_intercept=True, flavor='lm')

    condmeantimeblocked_q_lm_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_q_lm', 'obs', 'condmeantimeblocked',
                              X_obs_q, y_obs_condmeantimeblocked,
                                                   scale=False, fit_intercept=True, flavor='lm')

    condmeantimeblocked_noq_lm_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_noq_lm', 'obs', 'condmeantimeblocked',
                              X_obs_noq, y_obs_condmeantimeblocked,
                                                   scale=False, fit_intercept=True, flavor='lm')

    # LassoCV (lassocv)


    probblocked_basicq_lassocv_results = \
        crossval_summarize_mm('obs_probblocked_basicq_lassocv', 'obs', 'probblocked',
                              X_obs_basicq, y_obs_probblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    probblocked_q_lassocv_results = \
        crossval_summarize_mm('obs_probblocked_q_lassocv', 'obs', 'probblocked',
                              X_obs_q, y_obs_probblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    probblocked_noq_lassocv_results = \
        crossval_summarize_mm('obs_probblocked_noq_lassocv', 'obs', 'probblocked',
                              X_obs_noq, y_obs_probblocked,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)


    condmeantimeblocked_basicq_lassocv_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_basicq_lassocv', 'obs', 'condmeantimeblocked',
                              X_obs_basicq, y_obs_condmeantimeblocked,
                                                   scale=True, flavor='lassocv', lasso_max_iter=3000)

    condmeantimeblocked_q_lassocv_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_q_lassocv', 'obs', 'condmeantimeblocked',
                              X_obs_q, y_obs_condmeantimeblocked,
                                                   scale=True, flavor='lassocv', lasso_max_iter=3000)

    condmeantimeblocked_noq_lassocv_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_noq_lassocv', 'obs', 'condmeantimeblocked',
                              X_obs_noq, y_obs_condmeantimeblocked,
                                                   scale=True, flavor='lassocv', lasso_max_iter=3000)


    # Polynomial regression (poly)


    probblocked_basicq_poly_results = \
        crossval_summarize_mm('obs_probblocked_basicq_poly', 'obs', 'probblocked',
                              X_obs_basicq, y_obs_probblocked,
                              scale=False, flavor='poly')

    probblocked_q_poly_results = \
        crossval_summarize_mm('obs_probblocked_q_poly', 'obs', 'probblocked',
                              X_obs_q, y_obs_probblocked,
                              scale=False, flavor='poly')

    probblocked_noq_poly_results = \
        crossval_summarize_mm('obs_probblocked_noq_poly', 'obs', 'probblocked',
                              X_obs_noq, y_obs_probblocked,
                              scale=False, flavor='poly')

    condmeantimeblocked_basicq_poly_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_basicq_poly', 'obs', 'condmeantimeblocked',
                              X_obs_basicq, y_obs_condmeantimeblocked,
                                                   scale=False, flavor='poly')

    condmeantimeblocked_q_poly_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_q_poly', 'obs', 'condmeantimeblocked',
                              X_obs_q, y_obs_condmeantimeblocked,
                                                   scale=False, flavor='poly')

    condmeantimeblocked_noq_poly_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_noq_poly', 'obs', 'condmeantimeblocked',
                              X_obs_noq, y_obs_condmeantimeblocked,
                                                   scale=False, flavor='poly')

    # Random forest (rf)


    probblocked_basicq_rf_results = \
        crossval_summarize_mm('obs_probblocked_basicq_rf', 'obs', 'probblocked',
                              X_obs_basicq, y_obs_probblocked,
                              scale=False, flavor='rf')

    probblocked_q_rf_results = \
        crossval_summarize_mm('obs_probblocked_q_rf', 'obs', 'probblocked',
                              X_obs_q, y_obs_probblocked,
                              scale=False, flavor='rf')

    probblocked_noq_rf_results = \
        crossval_summarize_mm('obs_probblocked_noq_rf', 'obs', 'probblocked',
                              X_obs_noq, y_obs_probblocked,
                              scale=False, flavor='rf')

    condmeantimeblocked_basicq_rf_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_basicq_rf', 'obs', 'condmeantimeblocked',
                              X_obs_basicq, y_obs_condmeantimeblocked,
                                                   scale=False, flavor='rf')

    condmeantimeblocked_q_rf_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_q_rf', 'obs', 'condmeantimeblocked',
                              X_obs_q, y_obs_condmeantimeblocked,
                                                   scale=False, flavor='rf')

    condmeantimeblocked_noq_rf_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_noq_rf', 'obs', 'condmeantimeblocked',
                              X_obs_noq, y_obs_condmeantimeblocked,
                                                   scale=False, flavor='rf')

    # Support vector regression (svr)


    probblocked_basicq_svr_results = \
        crossval_summarize_mm('obs_probblocked_basicq_svr', 'obs', 'probblocked',
                              X_obs_basicq, y_obs_probblocked,
                              scale=True, flavor='svr')

    probblocked_q_svr_results = \
        crossval_summarize_mm('obs_probblocked_q_svr', 'obs', 'probblocked',
                              X_obs_q, y_obs_probblocked,
                              scale=True, flavor='svr')

    probblocked_noq_svr_results = \
        crossval_summarize_mm('obs_probblocked_noq_svr', 'obs', 'probblocked',
                              X_obs_noq, y_obs_probblocked,
                              scale=True, flavor='svr')

    condmeantimeblocked_basicq_svr_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_basicq_svr', 'obs', 'condmeantimeblocked',
                              X_obs_basicq, y_obs_condmeantimeblocked,
                                                   scale=True, flavor='svr')

    condmeantimeblocked_q_svr_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_q_svr', 'obs', 'condmeantimeblocked',
                              X_obs_q, y_obs_condmeantimeblocked,
                                                   scale=True, flavor='svr')

    condmeantimeblocked_noq_svr_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_noq_svr', 'obs', 'condmeantimeblocked',
                              X_obs_noq, y_obs_condmeantimeblocked,
                                                   scale=True, flavor='svr')

    # MLPRegressor Neural net (nn)


    probblocked_basicq_nn_results = \
        crossval_summarize_mm('obs_probblocked_basicq_nn', 'obs', 'probblocked',
                              X_obs_basicq, y_obs_probblocked,
                              scale=True, flavor='nn')

    probblocked_q_nn_results = \
        crossval_summarize_mm('obs_probblocked_q_nn', 'obs', 'probblocked',
                              X_obs_q, y_obs_probblocked,
                              scale=True, flavor='nn')

    probblocked_noq_nn_results = \
        crossval_summarize_mm('obs_probblocked_noq_nn', 'obs', 'probblocked',
                              X_obs_noq, y_obs_probblocked,
                              scale=True, flavor='nn')

    condmeantimeblocked_basicq_nn_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_basicq_nn', 'obs', 'condmeantimeblocked',
                              X_obs_basicq, y_obs_condmeantimeblocked,
                                                   scale=True, flavor='nn')

    condmeantimeblocked_q_nn_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_q_nn', 'obs', 'condmeantimeblocked',
                              X_obs_q, y_obs_condmeantimeblocked,
                                                   scale=True, flavor='nn')

    condmeantimeblocked_noq_nn_results = \
        crossval_summarize_mm('obs_condmeantimeblocked_noq_nn', 'obs', 'condmeantimeblocked',
                              X_obs_noq, y_obs_condmeantimeblocked,
                                                   scale=True, flavor='nn')

    obs_results = {'obs_occmean_basicq_lm_results': obs_occmean_basicq_lm_results,
                   'obs_occmean_q_lm_results': obs_occmean_q_lm_results,
                   'obs_occmean_noq_lm_results': obs_occmean_noq_lm_results,
                   'obs_occmean_basicq_lassocv_results': obs_occmean_basicq_lassocv_results,
                   'obs_occmean_q_lassocv_results': obs_occmean_q_lassocv_results,
                   'obs_occmean_noq_lassocv_results': obs_occmean_noq_lassocv_results,
                   'obs_occmean_basicq_poly_results': obs_occmean_basicq_poly_results,
                   'obs_occmean_q_poly_results': obs_occmean_q_poly_results,
                   'obs_occmean_noq_poly_results': obs_occmean_noq_poly_results,
                   'obs_occmean_basicq_rf_results': obs_occmean_basicq_rf_results,
                   'obs_occmean_q_rf_results': obs_occmean_q_rf_results,
                   'obs_occmean_noq_rf_results': obs_occmean_noq_rf_results,
                   'obs_occmean_basicq_svr_results': obs_occmean_basicq_svr_results,
                   'obs_occmean_q_svr_results': obs_occmean_q_svr_results,
                   'obs_occmean_noq_svr_results': obs_occmean_noq_svr_results,
                   'obs_occmean_basicq_nn_results': obs_occmean_basicq_nn_results,
                   'obs_occmean_q_nn_results': obs_occmean_q_nn_results,
                   'obs_occmean_noq_nn_results': obs_occmean_noq_nn_results,
                   'obs_occp95_basicq_lm_results': obs_occp95_basicq_lm_results,
                   'obs_occp95_q_lm_results': obs_occp95_q_lm_results,
                   'obs_occp95_noq_lm_results': obs_occp95_noq_lm_results,
                   'obs_occp95_basicq_lassocv_results': obs_occp95_basicq_lassocv_results,
                   'obs_occp95_q_lassocv_results': obs_occp95_q_lassocv_results,
                   'obs_occp95_noq_lassocv_results': obs_occp95_noq_lassocv_results,
                   'obs_occp95_basicq_poly_results': obs_occp95_basicq_poly_results,
                   'obs_occp95_q_poly_results': obs_occp95_q_poly_results,
                   'obs_occp95_noq_poly_results': obs_occp95_noq_poly_results,
                   'obs_occp95_basicq_rf_results': obs_occp95_basicq_rf_results,
                   'obs_occp95_q_rf_results': obs_occp95_q_rf_results,
                   'obs_occp95_noq_rf_results': obs_occp95_noq_rf_results,
                   'obs_occp95_basicq_svr_results': obs_occp95_basicq_svr_results,
                   'obs_occp95_q_svr_results': obs_occp95_q_svr_results,
                   'obs_occp95_noq_svr_results': obs_occp95_noq_svr_results,
                   'obs_occp95_basicq_nn_results': obs_occp95_basicq_nn_results,
                   'obs_occp95_q_nn_results': obs_occp95_q_nn_results,
                   'obs_occp95_noq_nn_results': obs_occp95_noq_nn_results,
                   'obs_probblocked_basicq_lm_results': probblocked_basicq_lm_results,
                   'obs_probblocked_q_lm_results': probblocked_q_lm_results,
                   'obs_probblocked_noq_lm_results': probblocked_noq_lm_results,
                   'obs_probblocked_basicq_lassocv_results': probblocked_basicq_lassocv_results,
                   'obs_probblocked_q_lassocv_results': probblocked_q_lassocv_results,
                   'obs_probblocked_noq_lassocv_results': probblocked_noq_lassocv_results,
                   'obs_probblocked_basicq_poly_results': probblocked_basicq_poly_results,
                   'obs_probblocked_q_poly_results': probblocked_q_poly_results,
                   'obs_probblocked_noq_poly_results': probblocked_noq_poly_results,
                   'obs_probblocked_basicq_rf_results': probblocked_basicq_rf_results,
                   'obs_probblocked_q_rf_results': probblocked_q_rf_results,
                   'obs_probblocked_noq_rf_results': probblocked_noq_rf_results,
                   'obs_probblocked_basicq_svr_results': probblocked_basicq_svr_results,
                   'obs_probblocked_q_svr_results': probblocked_q_svr_results,
                   'obs_probblocked_noq_svr_results': probblocked_noq_svr_results,
                   'obs_probblocked_basicq_nn_results': probblocked_basicq_nn_results,
                   'obs_probblocked_q_nn_results': probblocked_q_nn_results,
                   'obs_probblocked_noq_nn_results': probblocked_noq_nn_results,
                   'obs_condmeantimeblocked_basicq_lm_results': condmeantimeblocked_basicq_lm_results,
                   'obs_condmeantimeblocked_q_lm_results': condmeantimeblocked_q_lm_results,
                   'obs_condmeantimeblocked_noq_lm_results': condmeantimeblocked_noq_lm_results,
                   'obs_condmeantimeblocked_basicq_lassocv_results': condmeantimeblocked_basicq_lassocv_results,
                   'obs_condmeantimeblocked_q_lassocv_results': condmeantimeblocked_q_lassocv_results,
                   'obs_condmeantimeblocked_noq_lassocv_results': condmeantimeblocked_noq_lassocv_results,
                   'obs_condmeantimeblocked_basicq_poly_results': condmeantimeblocked_basicq_poly_results,
                   'obs_condmeantimeblocked_q_poly_results': condmeantimeblocked_q_poly_results,
                   'obs_condmeantimeblocked_noq_poly_results': condmeantimeblocked_noq_poly_results,
                   'obs_condmeantimeblocked_basicq_rf_results': condmeantimeblocked_basicq_rf_results,
                   'obs_condmeantimeblocked_q_rf_results': condmeantimeblocked_q_rf_results,
                   'obs_condmeantimeblocked_noq_rf_results': condmeantimeblocked_noq_rf_results,
                   'obs_condmeantimeblocked_basicq_svr_results': condmeantimeblocked_basicq_svr_results,
                   'obs_condmeantimeblocked_q_svr_results': condmeantimeblocked_q_svr_results,
                   'obs_condmeantimeblocked_noq_svr_results': condmeantimeblocked_noq_svr_results,
                   'obs_condmeantimeblocked_basicq_nn_results': condmeantimeblocked_basicq_nn_results,
                   'obs_condmeantimeblocked_q_nn_results': condmeantimeblocked_q_nn_results,
                   'obs_condmeantimeblocked_noq_nn_results': condmeantimeblocked_noq_nn_results,
                   'obs_occmean_q_load_results': obs_occmean_q_load_results,
                   'obs_occp95_q_sqrtload_results': obs_occp95_q_sqrtload_results,
                   'obs_probblocked_q_erlangc_results': probblocked_q_erlangc_results,
                   'obs_condmeantimeblocked_q_mgc_results': condmeantimeblocked_q_mgc_results,
                   'obs_occmean_onlyq_lm_results': obs_occmean_onlyq_lm_results,
                   'obs_occp95_onlyq_lm_results': obs_occp95_onlyq_lm_results,
                   'obs_probblocked_onlyq_lm_results': probblocked_onlyq_lm_results,
                   'obs_condmeantimeblocked_onlyq_lm_results': condmeantimeblocked_onlyq_lm_results,
                   }

    create_cv_plots(experiment, unit, obs_results, figures_path)
    create_coeff_plots(experiment, unit, obs_results, figures_path)

    metrics_df = create_metrics_df(obs_results)
    metrics_df.to_csv(metrics_path_filename, index=False)

    predictions_df = create_predictions_df(obs_results)
    predictions_df.to_csv(Path(output_path, f"{experiment}_{unit}_predictions.csv"), index=False)

    sys.setrecursionlimit(10000)
    # Pickle the results
    with open(pickle_path_filename, 'wb') as persisted_file:
        pickle.dump(obs_results, persisted_file)


def process_command_line(argv=None):
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='mm_run_fits_obs',
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
