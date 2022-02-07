from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from mmfitting import crossval_summarize_mm

experiment = "exp11"
data_path = Path("data")
output_path = Path("output")
figures_path = Path("output", "figures")
raw_data_path = Path("data", "raw")
pickle_filename = f"obs_results_{experiment}.pkl"

# X matrices
X_obs_noq = pd.read_csv(Path(data_path, f'X_obs_noq_{experiment}.csv'), index_col=0)
X_obs_basicq = pd.read_csv(Path(data_path, f'X_obs_basicq_{experiment}.csv'), index_col=0)
X_obs_q = pd.read_csv(Path(data_path, f'X_obs_q_{experiment}.csv'), index_col=0)

X_obs_occ_mean_onlyq = pd.read_csv(Path(data_path, f'X_obs_occmean_onlyq_{experiment}.csv'), index_col=0)
X_obs_occ_p95_onlyq = pd.read_csv(Path(data_path, f'X_obs_occp95_onlyq_{experiment}.csv'), index_col=0)
X_obs_prob_blockedby_ldr_onlyq = pd.read_csv(Path(data_path, f'X_obs_prob_blockedby_ldr_onlyq_{experiment}.csv'), index_col=0)
X_obs_condmeantime_blockedby_ldr_onlyq = \
    pd.read_csv(Path(data_path, f'X_obs_condmeantime_blockedby_ldr_onlyq_{experiment}.csv'), index_col=0)

# y vectors
y_obs_occ_mean = pd.read_csv(Path(data_path, f'y_obs_occ_mean_{experiment}.csv'), index_col=0, squeeze=True)
y_obs_occ_p95 = pd.read_csv(Path(data_path, f'y_obs_occ_p95_{experiment}.csv'), index_col=0, squeeze=True)
y_prob_blockedby_ldr = pd.read_csv(Path(data_path,
                                             f'y_prob_blockedby_ldr_{experiment}.csv'), index_col=0, squeeze=True)
y_condmeantime_blockedby_ldr = pd.read_csv(Path(data_path,
                                               f'y_condmeantime_blockedby_ldr_{experiment}.csv'),
                                          index_col=0, squeeze=True)

# Queueing models

obs_occ_mean_q_load_results = \
    crossval_summarize_mm('obs_occ_mean_q_load', 'obs', 'occ_mean', X_obs_q, y_obs_occ_mean, scale=False,
                          flavor='load', col_idx_arate=0, col_idx_meansvctime=1)

obs_occ_p95_q_sqrtload_results = \
    crossval_summarize_mm('obs_occ_p95_q_sqrtload', 'obs', 'occ_p95', X_obs_q, y_obs_occ_p95, scale=False,
                          flavor='sqrtload', col_idx_arate=0, col_idx_meansvctime=1, load_pctile=0.95)

obs_occ_mean_q_effload_results = \
    crossval_summarize_mm('obs_occ_mean_q_effload', 'obs', 'occ_mean', X_obs_q, y_obs_occ_mean, scale=False,
                          flavor='load', col_idx_arate=0, col_idx_meansvctime=19)

obs_occ_p95_q_sqrteffload_results = \
    crossval_summarize_mm('obs_occ_p95_q_sqrteffload', 'obs', 'occ_p95', X_obs_q, y_obs_occ_p95, scale=False,
                          flavor='sqrtload', col_idx_arate=0, col_idx_meansvctime=19, load_pctile=0.95)

prob_blockedby_ldr_q_erlangc_results = \
    crossval_summarize_mm('prob_blockedby_ldr_q_erlangc', 'obs', 'pct_blockedby_ldr',
                          X_obs_q, y_prob_blockedby_ldr,
                          scale=False, fit_intercept=True,
                          flavor='erlangc', col_idx_arate=0, col_idx_meansvctime=21, col_idx_numservers=4)

condmeantime_blockedby_ldr_q_mgc_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_q_mgc', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_q, y_condmeantime_blockedby_ldr,
                          scale=False, fit_intercept=True,
                          flavor='condmeanwaitldr', col_idx_arate=0, col_idx_meansvctime=21, col_idx_numservers=4,
                          col_idx_cv2svctime=18)

# Linear models using only queueing approximation terms
obs_occ_mean_onlyq_lm_results = \
    crossval_summarize_mm('obs_occ_mean_onlyq_lm', 'obs', 'occ_mean',
                          X_obs_occ_mean_onlyq, y_obs_occ_mean, scale=False, flavor='lm')

obs_occ_p95_onlyq_lm_results = \
    crossval_summarize_mm('obs_occ_p95_onlyq_lm', 'obs', 'occ_p95',
                          X_obs_occ_p95_onlyq, y_obs_occ_p95, scale=False, flavor='lm')

prob_blockedby_ldr_onlyq_lm_results = \
    crossval_summarize_mm('prob_blockedby_ldr_onlyq_lm', 'obs', 'pct_blockedby_ldr',
                          X_obs_prob_blockedby_ldr_onlyq, y_prob_blockedby_ldr, scale=False, flavor='lm')

condmeantime_blockedby_ldr_onlyq_lm_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_onlyq_lm', 'ldr', 'condmeantime_blockedby_ldr',
                          X_obs_condmeantime_blockedby_ldr_onlyq, y_condmeantime_blockedby_ldr, scale=False, flavor='lm')

## Linear regression (lm)
obs_occ_mean_basicq_lm_results = \
    crossval_summarize_mm('obs_occ_mean_basicq_lm', 'obs', 'occ_mean',
                          X_obs_basicq, y_obs_occ_mean, scale=False, flavor='lm')

obs_occ_mean_q_lm_results = \
    crossval_summarize_mm('obs_occ_mean_q_lm', 'obs', 'occ_mean',
                          X_obs_q, y_obs_occ_mean, scale=False, flavor='lm')

obs_occ_mean_noq_lm_results = \
    crossval_summarize_mm('obs_occ_mean_noq_lm', 'obs', 'occ_mean',
                          X_obs_noq, y_obs_occ_mean, scale=False, flavor='lm')


obs_occ_p95_basicq_lm_results = \
    crossval_summarize_mm('obs_occ_p95_basicq_lm', 'obs', 'occ_p95',
                          X_obs_basicq, y_obs_occ_p95, scale=False, flavor='lm')

obs_occ_p95_q_lm_results = \
    crossval_summarize_mm('obs_occ_p95_q_lm', 'obs', 'occ_p95',
                          X_obs_q, y_obs_occ_p95, scale=False, flavor='lm')

obs_occ_p95_noq_lm_results = \
    crossval_summarize_mm('obs_occ_p95_noq_lm', 'obs', 'occ_p95',
                          X_obs_noq, y_obs_occ_p95, scale=False, flavor='lm')



# LassoCV (lassocv)
obs_occ_mean_basicq_lassocv_results = \
    crossval_summarize_mm('obs_occ_mean_basicq_lassocv', 'obs', 'occ_mean',
                          X_obs_basicq, y_obs_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

obs_occ_mean_q_lassocv_results = \
    crossval_summarize_mm('obs_occ_mean_q_lassocv', 'obs', 'occ_mean',
                          X_obs_q, y_obs_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

obs_occ_mean_noq_lassocv_results = \
    crossval_summarize_mm('obs_occ_mean_noq_lassocv', 'obs', 'occ_mean',
                          X_obs_noq, y_obs_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

obs_occ_p95_basicq_lassocv_results = \
    crossval_summarize_mm('obs_occ_p95_basicq_lassocv', 'obs', 'occ_p95',
                          X_obs_basicq, y_obs_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

obs_occ_p95_q_lassocv_results = \
    crossval_summarize_mm('obs_occ_p95_q_lassocv', 'obs', 'occ_p95',
                          X_obs_q, y_obs_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

obs_occ_p95_noq_lassocv_results = \
    crossval_summarize_mm('obs_occ_p95_noq_lassocv', 'obs', 'occ_p95',
                          X_obs_noq, y_obs_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)




# Polynomial regression (poly)
obs_occ_mean_basicq_poly_results = \
    crossval_summarize_mm('obs_occ_mean_basicq_poly', 'obs', 'occ_mean',
                          X_obs_basicq, y_obs_occ_mean, scale=False, flavor='poly')

obs_occ_mean_q_poly_results = \
    crossval_summarize_mm('obs_occ_mean_q_poly', 'obs', 'occ_mean',
                          X_obs_q, y_obs_occ_mean, scale=False, flavor='poly')

obs_occ_mean_noq_poly_results = \
    crossval_summarize_mm('obs_occ_mean_noq_poly', 'obs', 'occ_mean',
                          X_obs_noq, y_obs_occ_mean, scale=False, flavor='poly')


obs_occ_p95_basicq_poly_results = \
    crossval_summarize_mm('obs_occ_p95_basicq_poly', 'obs', 'occ_p95',
                          X_obs_basicq, y_obs_occ_p95, scale=False, flavor='poly')

obs_occ_p95_q_poly_results = \
    crossval_summarize_mm('obs_occ_p95_q_poly', 'obs', 'occ_p95',
                          X_obs_q, y_obs_occ_p95, scale=False, flavor='poly')

obs_occ_p95_noq_poly_results = \
    crossval_summarize_mm('obs_occ_p95_noq_poly', 'obs', 'occ_p95',
                          X_obs_noq, y_obs_occ_p95, scale=False, flavor='poly')



# Random forest (rf)
obs_occ_mean_basicq_rf_results = \
    crossval_summarize_mm('obs_occ_mean_basicq_rf', 'obs', 'occ_mean',
                          X_obs_basicq, y_obs_occ_mean, scale=False, flavor='rf')

obs_occ_mean_q_rf_results = \
    crossval_summarize_mm('obs_occ_mean_q_rf', 'obs', 'occ_mean',
                          X_obs_q, y_obs_occ_mean, scale=False, flavor='rf')

obs_occ_mean_noq_rf_results = \
    crossval_summarize_mm('obs_occ_mean_noq_rf', 'obs', 'occ_mean',
                          X_obs_noq, y_obs_occ_mean, scale=False, flavor='rf')

obs_occ_p95_basicq_rf_results = \
    crossval_summarize_mm('obs_occ_p95_basicq_rf', 'obs', 'occ_p95',
                          X_obs_basicq, y_obs_occ_p95, scale=False, flavor='rf')

obs_occ_p95_q_rf_results = \
    crossval_summarize_mm('obs_occ_p95_q_rf', 'obs', 'occ_p95',
                          X_obs_q, y_obs_occ_p95, scale=False, flavor='rf')

obs_occ_p95_noq_rf_results = \
    crossval_summarize_mm('obs_occ_p95_noq_rf', 'obs', 'occ_p95',
                          X_obs_noq, y_obs_occ_p95, scale=False, flavor='rf')



# Support vector regression (svr)
obs_occ_mean_basicq_svr_results = \
    crossval_summarize_mm('obs_occ_mean_basicq_svr', 'obs', 'occ_mean',
                          X_obs_basicq, y_obs_occ_mean, flavor='svr', scale=True)

obs_occ_mean_q_svr_results = \
    crossval_summarize_mm('obs_occ_mean_q_svr', 'obs', 'occ_mean',
                          X_obs_q, y_obs_occ_mean, flavor='svr', scale=True)

obs_occ_mean_noq_svr_results = \
    crossval_summarize_mm('obs_occ_mean_noq_svr', 'obs', 'occ_mean',
                          X_obs_noq, y_obs_occ_mean, flavor='svr', scale=True)


obs_occ_p95_basicq_svr_results = \
    crossval_summarize_mm('obs_occ_p95_basicq_svr', 'obs', 'occ_p95',
                          X_obs_basicq, y_obs_occ_p95, flavor='svr', scale=True)

obs_occ_p95_q_svr_results = \
    crossval_summarize_mm('obs_occ_p95_q_svr', 'obs', 'occ_p95',
                          X_obs_q, y_obs_occ_p95, flavor='svr', scale=True)

obs_occ_p95_noq_svr_results = \
    crossval_summarize_mm('obs_occ_p95_noq_svr', 'obs', 'occ_p95',
                          X_obs_noq, y_obs_occ_p95, flavor='svr', scale=True)



# MLPRegressor Neural net (nn)
obs_occ_mean_basicq_nn_results = \
    crossval_summarize_mm('obs_occ_mean_basicq_nn', 'obs', 'occ_mean',
                          X_obs_basicq, y_obs_occ_mean, flavor='nn', scale=True)

obs_occ_mean_q_nn_results = \
    crossval_summarize_mm('obs_occ_mean_q_nn', 'obs', 'occ_mean',
                          X_obs_q, y_obs_occ_mean, flavor='nn', scale=True)

obs_occ_mean_noq_nn_results = \
    crossval_summarize_mm('obs_occ_mean_noq_nn', 'obs', 'occ_mean',
                          X_obs_noq, y_obs_occ_mean, flavor='nn', scale=True)

obs_occ_p95_basicq_nn_results = \
    crossval_summarize_mm('obs_occ_p95_basicq_nn', 'obs', 'occ_p95',
                          X_obs_basicq, y_obs_occ_p95, flavor='nn', scale=True)

obs_occ_p95_q_nn_results = \
    crossval_summarize_mm('obs_occ_p95_q_nn', 'obs', 'occ_p95',
                          X_obs_q, y_obs_occ_p95, flavor='nn', scale=True)

obs_occ_p95_noq_nn_results = \
    crossval_summarize_mm('obs_occ_p95_noq_nn', 'obs', 'occ_p95',
                          X_obs_noq, y_obs_occ_p95, flavor='nn', scale=True)


## Linear regression (lm)


prob_blockedby_ldr_basicq_lm_results = \
    crossval_summarize_mm('prob_blockedby_ldr_basicq_lm', 'obs', 'pct_blockedby_ldr',
                          X_obs_basicq, y_prob_blockedby_ldr,
                          scale=False, fit_intercept=True, flavor='lm')

prob_blockedby_ldr_q_lm_results = \
    crossval_summarize_mm('prob_blockedby_ldr_q_lm', 'obs', 'pct_blockedby_ldr',
                          X_obs_q, y_prob_blockedby_ldr,
                          scale=False, fit_intercept=True, flavor='lm')

prob_blockedby_ldr_noq_lm_results = \
    crossval_summarize_mm('prob_blockedby_ldr_noq_lm', 'obs', 'pct_blockedby_ldr',
                          X_obs_noq, y_prob_blockedby_ldr,
                          scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedby_ldr_basicq_lm_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_basicq_lm', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_basicq, y_condmeantime_blockedby_ldr,
                                               scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedby_ldr_q_lm_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_q_lm', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_q, y_condmeantime_blockedby_ldr,
                                               scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedby_ldr_noq_lm_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_noq_lm', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_noq, y_condmeantime_blockedby_ldr,
                                               scale=False, fit_intercept=True, flavor='lm')

# LassoCV (lassocv)


prob_blockedby_ldr_basicq_lassocv_results = \
    crossval_summarize_mm('prob_blockedby_ldr_basicq_lassocv', 'obs', 'pct_blockedby_ldr',
                          X_obs_basicq, y_prob_blockedby_ldr,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

prob_blockedby_ldr_q_lassocv_results = \
    crossval_summarize_mm('prob_blockedby_ldr_q_lassocv', 'obs', 'pct_blockedby_ldr',
                          X_obs_q, y_prob_blockedby_ldr,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

prob_blockedby_ldr_noq_lassocv_results = \
    crossval_summarize_mm('prob_blockedby_ldr_noq_lassocv', 'obs', 'pct_blockedby_ldr',
                          X_obs_noq, y_prob_blockedby_ldr,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)


condmeantime_blockedby_ldr_basicq_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_basicq_lassocv', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_basicq, y_condmeantime_blockedby_ldr,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

condmeantime_blockedby_ldr_q_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_q_lassocv', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_q, y_condmeantime_blockedby_ldr,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

condmeantime_blockedby_ldr_noq_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_noq_lassocv', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_noq, y_condmeantime_blockedby_ldr,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)


# Polynomial regression (poly)


prob_blockedby_ldr_basicq_poly_results = \
    crossval_summarize_mm('prob_blockedby_ldr_basicq_poly', 'obs', 'pct_blockedby_ldr',
                          X_obs_basicq, y_prob_blockedby_ldr,
                          scale=False, flavor='poly')

prob_blockedby_ldr_q_poly_results = \
    crossval_summarize_mm('prob_blockedby_ldr_q_poly', 'obs', 'pct_blockedby_ldr',
                          X_obs_q, y_prob_blockedby_ldr,
                          scale=False, flavor='poly')

prob_blockedby_ldr_noq_poly_results = \
    crossval_summarize_mm('prob_blockedby_ldr_noq_poly', 'obs', 'pct_blockedby_ldr',
                          X_obs_noq, y_prob_blockedby_ldr,
                          scale=False, flavor='poly')

condmeantime_blockedby_ldr_basicq_poly_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_basicq_poly', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_basicq, y_condmeantime_blockedby_ldr,
                                               scale=False, flavor='poly')

condmeantime_blockedby_ldr_q_poly_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_q_poly', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_q, y_condmeantime_blockedby_ldr,
                                               scale=False, flavor='poly')

condmeantime_blockedby_ldr_noq_poly_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_noq_poly', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_noq, y_condmeantime_blockedby_ldr,
                                               scale=False, flavor='poly')

# Random forest (rf)


prob_blockedby_ldr_basicq_rf_results = \
    crossval_summarize_mm('prob_blockedby_ldr_basicq_rf', 'obs', 'pct_blockedby_ldr',
                          X_obs_basicq, y_prob_blockedby_ldr,
                          scale=False, flavor='rf')

prob_blockedby_ldr_q_rf_results = \
    crossval_summarize_mm('prob_blockedby_ldr_q_rf', 'obs', 'pct_blockedby_ldr',
                          X_obs_q, y_prob_blockedby_ldr,
                          scale=False, flavor='rf')

prob_blockedby_ldr_noq_rf_results = \
    crossval_summarize_mm('prob_blockedby_ldr_noq_rf', 'obs', 'pct_blockedby_ldr',
                          X_obs_noq, y_prob_blockedby_ldr,
                          scale=False, flavor='rf')

condmeantime_blockedby_ldr_basicq_rf_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_basicq_rf', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_basicq, y_condmeantime_blockedby_ldr,
                                               scale=False, flavor='rf')

condmeantime_blockedby_ldr_q_rf_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_q_rf', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_q, y_condmeantime_blockedby_ldr,
                                               scale=False, flavor='rf')

condmeantime_blockedby_ldr_noq_rf_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_noq_rf', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_noq, y_condmeantime_blockedby_ldr,
                                               scale=False, flavor='rf')

# Support vector regression (svr)


prob_blockedby_ldr_basicq_svr_results = \
    crossval_summarize_mm('prob_blockedby_ldr_basicq_svr', 'obs', 'pct_blockedby_ldr',
                          X_obs_basicq, y_prob_blockedby_ldr,
                          scale=True, flavor='svr')

prob_blockedby_ldr_q_svr_results = \
    crossval_summarize_mm('prob_blockedby_ldr_q_svr', 'obs', 'pct_blockedby_ldr',
                          X_obs_q, y_prob_blockedby_ldr,
                          scale=True, flavor='svr')

prob_blockedby_ldr_noq_svr_results = \
    crossval_summarize_mm('prob_blockedby_ldr_noq_svr', 'obs', 'pct_blockedby_ldr',
                          X_obs_noq, y_prob_blockedby_ldr,
                          scale=True, flavor='svr')

condmeantime_blockedby_ldr_basicq_svr_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_basicq_svr', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_basicq, y_condmeantime_blockedby_ldr,
                                               scale=True, flavor='svr')

condmeantime_blockedby_ldr_q_svr_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_q_svr', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_q, y_condmeantime_blockedby_ldr,
                                               scale=True, flavor='svr')

condmeantime_blockedby_ldr_noq_svr_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_noq_svr', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_noq, y_condmeantime_blockedby_ldr,
                                               scale=True, flavor='svr')

# MLPRegressor Neural net (nn)


prob_blockedby_ldr_basicq_nn_results = \
    crossval_summarize_mm('prob_blockedby_ldr_basicq_nn', 'obs', 'pct_blockedby_ldr',
                          X_obs_basicq, y_prob_blockedby_ldr,
                          scale=True, flavor='nn')

prob_blockedby_ldr_q_nn_results = \
    crossval_summarize_mm('prob_blockedby_ldr_q_nn', 'obs', 'pct_blockedby_ldr',
                          X_obs_q, y_prob_blockedby_ldr,
                          scale=True, flavor='nn')

prob_blockedby_ldr_noq_nn_results = \
    crossval_summarize_mm('prob_blockedby_ldr_noq_nn', 'obs', 'pct_blockedby_ldr',
                          X_obs_noq, y_prob_blockedby_ldr,
                          scale=True, flavor='nn')

condmeantime_blockedby_ldr_basicq_nn_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_basicq_nn', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_basicq, y_condmeantime_blockedby_ldr,
                                               scale=True, flavor='nn')

condmeantime_blockedby_ldr_q_nn_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_q_nn', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_q, y_condmeantime_blockedby_ldr,
                                               scale=True, flavor='nn')

condmeantime_blockedby_ldr_noq_nn_results = \
    crossval_summarize_mm('condmeantime_blockedby_ldr_noq_nn', 'obs', 'condmeantime_blockedby_ldr',
                          X_obs_noq, y_condmeantime_blockedby_ldr,
                                               scale=True, flavor='nn')

obs_results = {'obs_occ_mean_basicq_lm_results': obs_occ_mean_basicq_lm_results,
               'obs_occ_mean_q_lm_results': obs_occ_mean_q_lm_results,
               'obs_occ_mean_noq_lm_results': obs_occ_mean_noq_lm_results,
               'obs_occ_mean_basicq_lassocv_results': obs_occ_mean_basicq_lassocv_results,
               'obs_occ_mean_q_lassocv_results': obs_occ_mean_q_lassocv_results,
               'obs_occ_mean_noq_lassocv_results': obs_occ_mean_noq_lassocv_results,
               'obs_occ_mean_basicq_poly_results': obs_occ_mean_basicq_poly_results,
               'obs_occ_mean_q_poly_results': obs_occ_mean_q_poly_results,
               'obs_occ_mean_noq_poly_results': obs_occ_mean_noq_poly_results,
               'obs_occ_mean_basicq_rf_results': obs_occ_mean_basicq_rf_results,
               'obs_occ_mean_q_rf_results': obs_occ_mean_q_rf_results,
               'obs_occ_mean_noq_rf_results': obs_occ_mean_noq_rf_results,
               'obs_occ_mean_basicq_svr_results': obs_occ_mean_basicq_svr_results,
               'obs_occ_mean_q_svr_results': obs_occ_mean_q_svr_results,
               'obs_occ_mean_noq_svr_results': obs_occ_mean_noq_svr_results,
               'obs_occ_mean_basicq_nn_results': obs_occ_mean_basicq_nn_results,
               'obs_occ_mean_q_nn_results': obs_occ_mean_q_nn_results,
               'obs_occ_mean_noq_nn_results': obs_occ_mean_noq_nn_results,
               'obs_occ_p95_basicq_lm_results': obs_occ_p95_basicq_lm_results,
               'obs_occ_p95_q_lm_results': obs_occ_p95_q_lm_results,
               'obs_occ_p95_noq_lm_results': obs_occ_p95_noq_lm_results,
               'obs_occ_p95_basicq_lassocv_results': obs_occ_p95_basicq_lassocv_results,
               'obs_occ_p95_q_lassocv_results': obs_occ_p95_q_lassocv_results,
               'obs_occ_p95_noq_lassocv_results': obs_occ_p95_noq_lassocv_results,
               'obs_occ_p95_basicq_poly_results': obs_occ_p95_basicq_poly_results,
               'obs_occ_p95_q_poly_results': obs_occ_p95_q_poly_results,
               'obs_occ_p95_noq_poly_results': obs_occ_p95_noq_poly_results,
               'obs_occ_p95_basicq_rf_results': obs_occ_p95_basicq_rf_results,
               'obs_occ_p95_q_rf_results': obs_occ_p95_q_rf_results,
               'obs_occ_p95_noq_rf_results': obs_occ_p95_noq_rf_results,
               'obs_occ_p95_basicq_svr_results': obs_occ_p95_basicq_svr_results,
               'obs_occ_p95_q_svr_results': obs_occ_p95_q_svr_results,
               'obs_occ_p95_noq_svr_results': obs_occ_p95_noq_svr_results,
               'obs_occ_p95_basicq_nn_results': obs_occ_p95_basicq_nn_results,
               'obs_occ_p95_q_nn_results': obs_occ_p95_q_nn_results,
               'obs_occ_p95_noq_nn_results': obs_occ_p95_noq_nn_results,
               'prob_blockedby_ldr_basicq_lm_results': prob_blockedby_ldr_basicq_lm_results,
               'prob_blockedby_ldr_q_lm_results': prob_blockedby_ldr_q_lm_results,
               'prob_blockedby_ldr_noq_lm_results': prob_blockedby_ldr_noq_lm_results,
               'prob_blockedby_ldr_basicq_lassocv_results': prob_blockedby_ldr_basicq_lassocv_results,
               'prob_blockedby_ldr_q_lassocv_results': prob_blockedby_ldr_q_lassocv_results,
               'prob_blockedby_ldr_noq_lassocv_results': prob_blockedby_ldr_noq_lassocv_results,
               'prob_blockedby_ldr_basicq_poly_results': prob_blockedby_ldr_basicq_poly_results,
               'prob_blockedby_ldr_q_poly_results': prob_blockedby_ldr_q_poly_results,
               'prob_blockedby_ldr_noq_poly_results': prob_blockedby_ldr_noq_poly_results,
               'prob_blockedby_ldr_basicq_rf_results': prob_blockedby_ldr_basicq_rf_results,
               'prob_blockedby_ldr_q_rf_results': prob_blockedby_ldr_q_rf_results,
               'prob_blockedby_ldr_noq_rf_results': prob_blockedby_ldr_noq_rf_results,
               'prob_blockedby_ldr_basicq_svr_results': prob_blockedby_ldr_basicq_svr_results,
               'prob_blockedby_ldr_q_svr_results': prob_blockedby_ldr_q_svr_results,
               'prob_blockedby_ldr_noq_svr_results': prob_blockedby_ldr_noq_svr_results,
               'prob_blockedby_ldr_basicq_nn_results': prob_blockedby_ldr_basicq_nn_results,
               'prob_blockedby_ldr_q_nn_results': prob_blockedby_ldr_q_nn_results,
               'prob_blockedby_ldr_noq_nn_results': prob_blockedby_ldr_noq_nn_results,
               'condmeantime_blockedby_ldr_basicq_lm_results': condmeantime_blockedby_ldr_basicq_lm_results,
               'condmeantime_blockedby_ldr_q_lm_results': condmeantime_blockedby_ldr_q_lm_results,
               'condmeantime_blockedby_ldr_noq_lm_results': condmeantime_blockedby_ldr_noq_lm_results,
               'condmeantime_blockedby_ldr_basicq_lassocv_results': condmeantime_blockedby_ldr_basicq_lassocv_results,
               'condmeantime_blockedby_ldr_q_lassocv_results': condmeantime_blockedby_ldr_q_lassocv_results,
               'condmeantime_blockedby_ldr_noq_lassocv_results': condmeantime_blockedby_ldr_noq_lassocv_results,
               'condmeantime_blockedby_ldr_basicq_poly_results': condmeantime_blockedby_ldr_basicq_poly_results,
               'condmeantime_blockedby_ldr_q_poly_results': condmeantime_blockedby_ldr_q_poly_results,
               'condmeantime_blockedby_ldr_noq_poly_results': condmeantime_blockedby_ldr_noq_poly_results,
               'condmeantime_blockedby_ldr_basicq_rf_results': condmeantime_blockedby_ldr_basicq_rf_results,
               'condmeantime_blockedby_ldr_q_rf_results': condmeantime_blockedby_ldr_q_rf_results,
               'condmeantime_blockedby_ldr_noq_rf_results': condmeantime_blockedby_ldr_noq_rf_results,
               'condmeantime_blockedby_ldr_basicq_svr_results': condmeantime_blockedby_ldr_basicq_svr_results,
               'condmeantime_blockedby_ldr_q_svr_results': condmeantime_blockedby_ldr_q_svr_results,
               'condmeantime_blockedby_ldr_noq_svr_results': condmeantime_blockedby_ldr_noq_svr_results,
               'condmeantime_blockedby_ldr_basicq_nn_results': condmeantime_blockedby_ldr_basicq_nn_results,
               'condmeantime_blockedby_ldr_q_nn_results': condmeantime_blockedby_ldr_q_nn_results,
               'condmeantime_blockedby_ldr_noq_nn_results': condmeantime_blockedby_ldr_noq_nn_results,
               'obs_occ_mean_q_load_results': obs_occ_mean_q_load_results,
               'obs_occ_p95_q_sqrtload_results': obs_occ_p95_q_sqrtload_results,
               'obs_occ_mean_q_effload_results': obs_occ_mean_q_effload_results,
               'obs_occ_p95_q_sqrteffload_results': obs_occ_p95_q_sqrteffload_results,
               'prob_blockedby_ldr_q_erlangc_results': prob_blockedby_ldr_q_erlangc_results,
               'condmeantime_blockedby_ldr_q_mgc_results': condmeantime_blockedby_ldr_q_mgc_results,
               'obs_occ_mean_onlyq_lm_results': obs_occ_mean_onlyq_lm_results,
               'obs_occ_p95_onlyq_lm_results': obs_occ_p95_onlyq_lm_results,
               'prob_blockedby_ldr_onlyq_lm_results': prob_blockedby_ldr_onlyq_lm_results,
               'condmeantime_blockedby_ldr_onlyq_lm_results': condmeantime_blockedby_ldr_onlyq_lm_results,
               }







# Pickle the results
with open(Path(output_path, pickle_filename), 'wb') as pickle_file:
    pickle.dump(obs_results, pickle_file)
