import sys
from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from obflowsim.mm.mm_fitting import crossval_summarize_mm
from obflowsim.mm.mm_process_fitted_models import create_cv_plots, create_coeff_plots, create_metrics_df


unit = "obs"
experiment = "exp11"
input_path = Path(f"input/{experiment}")
output_path = Path(f"output/{experiment}")
figures_path = Path(f"output/{experiment}", "figures")


metrics_path_filename = Path(output_path, f"{experiment}_{unit}_metrics.csv")
pickle_path_filename = Path(output_path, f"{experiment}_{unit}_results.pkl")

# X matrices
X_obs_noq = pd.read_csv(Path(input_path, f'X_obs_noq_{experiment}.csv'), index_col=0)
X_obs_basicq = pd.read_csv(Path(input_path, f'X_obs_basicq_{experiment}.csv'), index_col=0)
X_obs_q = pd.read_csv(Path(input_path, f'X_obs_q_{experiment}.csv'), index_col=0)

X_obs_occmean_onlyq = pd.read_csv(Path(input_path, f'X_obs_occmean_onlyq_{experiment}.csv'), index_col=0)
X_obs_occp95_onlyq = pd.read_csv(Path(input_path, f'X_obs_occp95_onlyq_{experiment}.csv'), index_col=0)
X_obs_probblockedbyldr_onlyq = pd.read_csv(Path(input_path, f'X_obs_probblockedbyldr_onlyq_{experiment}.csv'), index_col=0)
X_obs_condmeantimeblockedbyldr_onlyq = \
    pd.read_csv(Path(input_path, f'X_obs_condmeantimeblockedbyldr_onlyq_{experiment}.csv'), index_col=0)

# y vectors
y_obs_occmean = pd.read_csv(Path(input_path, f'y_obs_occmean_{experiment}.csv'), index_col=0, squeeze=True)
y_obs_occp95 = pd.read_csv(Path(input_path, f'y_obs_occp95_{experiment}.csv'), index_col=0, squeeze=True)
y_probblockedbyldr = pd.read_csv(Path(input_path,
                                             f'y_probblockedbyldr_{experiment}.csv'), index_col=0, squeeze=True)
y_condmeantimeblockedbyldr = pd.read_csv(Path(input_path,
                                               f'y_condmeantimeblockedbyldr_{experiment}.csv'),
                                          index_col=0, squeeze=True)

# Queueing models

obs_occmean_q_load_results = \
    crossval_summarize_mm('obs_occmean_q_load', 'obs', 'occmean', X_obs_q, y_obs_occmean, scale=False,
                          flavor='load', col_idx_arate=0, col_idx_meansvctime=1)

obs_occp95_q_sqrtload_results = \
    crossval_summarize_mm('obs_occp95_q_sqrtload', 'obs', 'occp95', X_obs_q, y_obs_occp95, scale=False,
                          flavor='sqrtload', col_idx_arate=0, col_idx_meansvctime=1, load_pctile=0.95)

obs_occmean_q_effload_results = \
    crossval_summarize_mm('obs_occmean_q_effload', 'obs', 'occmean', X_obs_q, y_obs_occmean, scale=False,
                          flavor='load', col_idx_arate=0, col_idx_meansvctime=19)

obs_occp95_q_sqrteffload_results = \
    crossval_summarize_mm('obs_occp95_q_sqrteffload', 'obs', 'occp95', X_obs_q, y_obs_occp95, scale=False,
                          flavor='sqrtload', col_idx_arate=0, col_idx_meansvctime=19, load_pctile=0.95)

probblockedbyldr_q_erlangc_results = \
    crossval_summarize_mm('obs_probblockedbyldr_q_erlangc', 'obs', 'probblockedbyldr',
                          X_obs_q, y_probblockedbyldr,
                          scale=False, fit_intercept=True,
                          flavor='erlangc', col_idx_arate=0, col_idx_meansvctime=21, col_idx_numservers=4)

condmeantimeblockedbyldr_q_mgc_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_q_mgc', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_q, y_condmeantimeblockedbyldr,
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

probblockedbyldr_onlyq_lm_results = \
    crossval_summarize_mm('obs_probblockedbyldr_onlyq_lm', 'obs', 'probblockedbyldr',
                          X_obs_probblockedbyldr_onlyq, y_probblockedbyldr, scale=False, flavor='lm')

condmeantimeblockedbyldr_onlyq_lm_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_onlyq_lm', 'ldr', 'condmeantimeblockedbyldr',
                          X_obs_condmeantimeblockedbyldr_onlyq, y_condmeantimeblockedbyldr, scale=False, flavor='lm')

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


probblockedbyldr_basicq_lm_results = \
    crossval_summarize_mm('obs_probblockedbyldr_basicq_lm', 'obs', 'probblockedbyldr',
                          X_obs_basicq, y_probblockedbyldr,
                          scale=False, fit_intercept=True, flavor='lm')

probblockedbyldr_q_lm_results = \
    crossval_summarize_mm('obs_probblockedbyldr_q_lm', 'obs', 'probblockedbyldr',
                          X_obs_q, y_probblockedbyldr,
                          scale=False, fit_intercept=True, flavor='lm')

probblockedbyldr_noq_lm_results = \
    crossval_summarize_mm('obs_probblockedbyldr_noq_lm', 'obs', 'probblockedbyldr',
                          X_obs_noq, y_probblockedbyldr,
                          scale=False, fit_intercept=True, flavor='lm')

condmeantimeblockedbyldr_basicq_lm_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_basicq_lm', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_basicq, y_condmeantimeblockedbyldr,
                                               scale=False, fit_intercept=True, flavor='lm')

condmeantimeblockedbyldr_q_lm_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_q_lm', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_q, y_condmeantimeblockedbyldr,
                                               scale=False, fit_intercept=True, flavor='lm')

condmeantimeblockedbyldr_noq_lm_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_noq_lm', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_noq, y_condmeantimeblockedbyldr,
                                               scale=False, fit_intercept=True, flavor='lm')

# LassoCV (lassocv)


probblockedbyldr_basicq_lassocv_results = \
    crossval_summarize_mm('obs_probblockedbyldr_basicq_lassocv', 'obs', 'probblockedbyldr',
                          X_obs_basicq, y_probblockedbyldr,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

probblockedbyldr_q_lassocv_results = \
    crossval_summarize_mm('obs_probblockedbyldr_q_lassocv', 'obs', 'probblockedbyldr',
                          X_obs_q, y_probblockedbyldr,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

probblockedbyldr_noq_lassocv_results = \
    crossval_summarize_mm('obs_probblockedbyldr_noq_lassocv', 'obs', 'probblockedbyldr',
                          X_obs_noq, y_probblockedbyldr,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)


condmeantimeblockedbyldr_basicq_lassocv_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_basicq_lassocv', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_basicq, y_condmeantimeblockedbyldr,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

condmeantimeblockedbyldr_q_lassocv_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_q_lassocv', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_q, y_condmeantimeblockedbyldr,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

condmeantimeblockedbyldr_noq_lassocv_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_noq_lassocv', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_noq, y_condmeantimeblockedbyldr,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)


# Polynomial regression (poly)


probblockedbyldr_basicq_poly_results = \
    crossval_summarize_mm('obs_probblockedbyldr_basicq_poly', 'obs', 'probblockedbyldr',
                          X_obs_basicq, y_probblockedbyldr,
                          scale=False, flavor='poly')

probblockedbyldr_q_poly_results = \
    crossval_summarize_mm('obs_probblockedbyldr_q_poly', 'obs', 'probblockedbyldr',
                          X_obs_q, y_probblockedbyldr,
                          scale=False, flavor='poly')

probblockedbyldr_noq_poly_results = \
    crossval_summarize_mm('obs_probblockedbyldr_noq_poly', 'obs', 'probblockedbyldr',
                          X_obs_noq, y_probblockedbyldr,
                          scale=False, flavor='poly')

condmeantimeblockedbyldr_basicq_poly_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_basicq_poly', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_basicq, y_condmeantimeblockedbyldr,
                                               scale=False, flavor='poly')

condmeantimeblockedbyldr_q_poly_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_q_poly', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_q, y_condmeantimeblockedbyldr,
                                               scale=False, flavor='poly')

condmeantimeblockedbyldr_noq_poly_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_noq_poly', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_noq, y_condmeantimeblockedbyldr,
                                               scale=False, flavor='poly')

# Random forest (rf)


probblockedbyldr_basicq_rf_results = \
    crossval_summarize_mm('obs_probblockedbyldr_basicq_rf', 'obs', 'probblockedbyldr',
                          X_obs_basicq, y_probblockedbyldr,
                          scale=False, flavor='rf')

probblockedbyldr_q_rf_results = \
    crossval_summarize_mm('obs_probblockedbyldr_q_rf', 'obs', 'probblockedbyldr',
                          X_obs_q, y_probblockedbyldr,
                          scale=False, flavor='rf')

probblockedbyldr_noq_rf_results = \
    crossval_summarize_mm('obs_probblockedbyldr_noq_rf', 'obs', 'probblockedbyldr',
                          X_obs_noq, y_probblockedbyldr,
                          scale=False, flavor='rf')

condmeantimeblockedbyldr_basicq_rf_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_basicq_rf', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_basicq, y_condmeantimeblockedbyldr,
                                               scale=False, flavor='rf')

condmeantimeblockedbyldr_q_rf_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_q_rf', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_q, y_condmeantimeblockedbyldr,
                                               scale=False, flavor='rf')

condmeantimeblockedbyldr_noq_rf_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_noq_rf', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_noq, y_condmeantimeblockedbyldr,
                                               scale=False, flavor='rf')

# Support vector regression (svr)


probblockedbyldr_basicq_svr_results = \
    crossval_summarize_mm('obs_probblockedbyldr_basicq_svr', 'obs', 'probblockedbyldr',
                          X_obs_basicq, y_probblockedbyldr,
                          scale=True, flavor='svr')

probblockedbyldr_q_svr_results = \
    crossval_summarize_mm('obs_probblockedbyldr_q_svr', 'obs', 'probblockedbyldr',
                          X_obs_q, y_probblockedbyldr,
                          scale=True, flavor='svr')

probblockedbyldr_noq_svr_results = \
    crossval_summarize_mm('obs_probblockedbyldr_noq_svr', 'obs', 'probblockedbyldr',
                          X_obs_noq, y_probblockedbyldr,
                          scale=True, flavor='svr')

condmeantimeblockedbyldr_basicq_svr_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_basicq_svr', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_basicq, y_condmeantimeblockedbyldr,
                                               scale=True, flavor='svr')

condmeantimeblockedbyldr_q_svr_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_q_svr', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_q, y_condmeantimeblockedbyldr,
                                               scale=True, flavor='svr')

condmeantimeblockedbyldr_noq_svr_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_noq_svr', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_noq, y_condmeantimeblockedbyldr,
                                               scale=True, flavor='svr')

# MLPRegressor Neural net (nn)


probblockedbyldr_basicq_nn_results = \
    crossval_summarize_mm('obs_probblockedbyldr_basicq_nn', 'obs', 'probblockedbyldr',
                          X_obs_basicq, y_probblockedbyldr,
                          scale=True, flavor='nn')

probblockedbyldr_q_nn_results = \
    crossval_summarize_mm('obs_probblockedbyldr_q_nn', 'obs', 'probblockedbyldr',
                          X_obs_q, y_probblockedbyldr,
                          scale=True, flavor='nn')

probblockedbyldr_noq_nn_results = \
    crossval_summarize_mm('obs_probblockedbyldr_noq_nn', 'obs', 'probblockedbyldr',
                          X_obs_noq, y_probblockedbyldr,
                          scale=True, flavor='nn')

condmeantimeblockedbyldr_basicq_nn_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_basicq_nn', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_basicq, y_condmeantimeblockedbyldr,
                                               scale=True, flavor='nn')

condmeantimeblockedbyldr_q_nn_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_q_nn', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_q, y_condmeantimeblockedbyldr,
                                               scale=True, flavor='nn')

condmeantimeblockedbyldr_noq_nn_results = \
    crossval_summarize_mm('obs_condmeantimeblockedbyldr_noq_nn', 'obs', 'condmeantimeblockedbyldr',
                          X_obs_noq, y_condmeantimeblockedbyldr,
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
               'obs_probblockedbyldr_basicq_lm_results': probblockedbyldr_basicq_lm_results,
               'obs_probblockedbyldr_q_lm_results': probblockedbyldr_q_lm_results,
               'obs_probblockedbyldr_noq_lm_results': probblockedbyldr_noq_lm_results,
               'obs_probblockedbyldr_basicq_lassocv_results': probblockedbyldr_basicq_lassocv_results,
               'obs_probblockedbyldr_q_lassocv_results': probblockedbyldr_q_lassocv_results,
               'obs_probblockedbyldr_noq_lassocv_results': probblockedbyldr_noq_lassocv_results,
               'obs_probblockedbyldr_basicq_poly_results': probblockedbyldr_basicq_poly_results,
               'obs_probblockedbyldr_q_poly_results': probblockedbyldr_q_poly_results,
               'obs_probblockedbyldr_noq_poly_results': probblockedbyldr_noq_poly_results,
               'obs_probblockedbyldr_basicq_rf_results': probblockedbyldr_basicq_rf_results,
               'obs_probblockedbyldr_q_rf_results': probblockedbyldr_q_rf_results,
               'obs_probblockedbyldr_noq_rf_results': probblockedbyldr_noq_rf_results,
               'obs_probblockedbyldr_basicq_svr_results': probblockedbyldr_basicq_svr_results,
               'obs_probblockedbyldr_q_svr_results': probblockedbyldr_q_svr_results,
               'obs_probblockedbyldr_noq_svr_results': probblockedbyldr_noq_svr_results,
               'obs_probblockedbyldr_basicq_nn_results': probblockedbyldr_basicq_nn_results,
               'obs_probblockedbyldr_q_nn_results': probblockedbyldr_q_nn_results,
               'obs_probblockedbyldr_noq_nn_results': probblockedbyldr_noq_nn_results,
               'obs_condmeantimeblockedbyldr_basicq_lm_results': condmeantimeblockedbyldr_basicq_lm_results,
               'obs_condmeantimeblockedbyldr_q_lm_results': condmeantimeblockedbyldr_q_lm_results,
               'obs_condmeantimeblockedbyldr_noq_lm_results': condmeantimeblockedbyldr_noq_lm_results,
               'obs_condmeantimeblockedbyldr_basicq_lassocv_results': condmeantimeblockedbyldr_basicq_lassocv_results,
               'obs_condmeantimeblockedbyldr_q_lassocv_results': condmeantimeblockedbyldr_q_lassocv_results,
               'obs_condmeantimeblockedbyldr_noq_lassocv_results': condmeantimeblockedbyldr_noq_lassocv_results,
               'obs_condmeantimeblockedbyldr_basicq_poly_results': condmeantimeblockedbyldr_basicq_poly_results,
               'obs_condmeantimeblockedbyldr_q_poly_results': condmeantimeblockedbyldr_q_poly_results,
               'obs_condmeantimeblockedbyldr_noq_poly_results': condmeantimeblockedbyldr_noq_poly_results,
               'obs_condmeantimeblockedbyldr_basicq_rf_results': condmeantimeblockedbyldr_basicq_rf_results,
               'obs_condmeantimeblockedbyldr_q_rf_results': condmeantimeblockedbyldr_q_rf_results,
               'obs_condmeantimeblockedbyldr_noq_rf_results': condmeantimeblockedbyldr_noq_rf_results,
               'obs_condmeantimeblockedbyldr_basicq_svr_results': condmeantimeblockedbyldr_basicq_svr_results,
               'obs_condmeantimeblockedbyldr_q_svr_results': condmeantimeblockedbyldr_q_svr_results,
               'obs_condmeantimeblockedbyldr_noq_svr_results': condmeantimeblockedbyldr_noq_svr_results,
               'obs_condmeantimeblockedbyldr_basicq_nn_results': condmeantimeblockedbyldr_basicq_nn_results,
               'obs_condmeantimeblockedbyldr_q_nn_results': condmeantimeblockedbyldr_q_nn_results,
               'obs_condmeantimeblockedbyldr_noq_nn_results': condmeantimeblockedbyldr_noq_nn_results,
               'obs_occmean_q_load_results': obs_occmean_q_load_results,
               'obs_occp95_q_sqrtload_results': obs_occp95_q_sqrtload_results,
               'obs_occmean_q_effload_results': obs_occmean_q_effload_results,
               'obs_occp95_q_sqrteffload_results': obs_occp95_q_sqrteffload_results,
               'obs_probblockedbyldr_q_erlangc_results': probblockedbyldr_q_erlangc_results,
               'obs_condmeantimeblockedbyldr_q_mgc_results': condmeantimeblockedbyldr_q_mgc_results,
               'obs_occmean_onlyq_lm_results': obs_occmean_onlyq_lm_results,
               'obs_occp95_onlyq_lm_results': obs_occp95_onlyq_lm_results,
               'obs_probblockedbyldr_onlyq_lm_results': probblockedbyldr_onlyq_lm_results,
               'obs_condmeantimeblockedbyldr_onlyq_lm_results': condmeantimeblockedbyldr_onlyq_lm_results,
               }

create_cv_plots(experiment, unit, obs_results, figures_path)
create_coeff_plots(experiment, unit, obs_results, figures_path)

metrics_df = create_metrics_df(obs_results)
metrics_df.to_csv(metrics_path_filename, index=False)


sys.setrecursionlimit(10000)
# Pickle the results
with open(pickle_path_filename, 'wb') as persisted_file:
    pickle.dump(obs_results, persisted_file)
