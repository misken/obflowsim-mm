from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from mmfitting import crossval_summarize_mm

plt.ioff()

experiment = "exp11"
data_path = Path("data")
output_path = Path("output")
figures_path = Path("output", "figures")
siminout_data_path = Path("data", "siminout")
pickle_filename = f"ldr_results_{experiment}.pkl"

# X matrices
X_ldr_noq = pd.read_csv(Path(data_path, f'X_ldr_noq_{experiment}.csv'), index_col=0)
X_ldr_basicq = pd.read_csv(Path(data_path, f'X_ldr_basicq_{experiment}.csv'), index_col=0)
X_ldr_q = pd.read_csv(Path(data_path, f'X_ldr_q_{experiment}.csv'), index_col=0)

X_ldr_occ_mean_onlyq = pd.read_csv(Path(data_path, f'X_ldr_occmean_onlyq_{experiment}.csv'), index_col=0)
X_ldr_occ_p95_onlyq = pd.read_csv(Path(data_path, f'X_ldr_occp95_onlyq_{experiment}.csv'), index_col=0)
X_ldr_prob_blockedby_pp_onlyq = pd.read_csv(Path(data_path, f'X_ldr_prob_blockedby_pp_onlyq_{experiment}.csv'),
                                            index_col=0)
X_ldr_condmeantime_blockedby_pp_onlyq = \
    pd.read_csv(Path(data_path, f'X_ldr_condmeantime_blockedby_pp_onlyq_{experiment}.csv'), index_col=0)

# y vectors
y_ldr_occ_mean = pd.read_csv(Path(data_path, f'y_ldr_occ_mean_{experiment}.csv'), index_col=0, squeeze=True)
y_ldr_occ_p95 = pd.read_csv(Path(data_path, f'y_ldr_occ_p95_{experiment}.csv'), index_col=0, squeeze=True)
y_prob_blockedby_pp = \
    pd.read_csv(Path(data_path, f'y_prob_blockedby_pp_{experiment}.csv'), index_col=0, squeeze=True)
y_condmeantime_blockedby_pp = \
    pd.read_csv(Path(data_path, f'y_condmeantime_blockedby_pp_{experiment}.csv'), index_col=0, squeeze=True)

# Fit models

# Queueing models
ldr_occ_mean_q_load_results = \
    crossval_summarize_mm('ldr_occ_mean_q_effload', 'ldr', 'occ_mean', X_ldr_q, y_ldr_occ_mean, scale=False,
                          flavor='load', col_idx_arate=0, col_idx_meansvctime=18)

ldr_occ_p95_q_sqrtload_results = \
    crossval_summarize_mm('ldr_occ_p95_q_sqrteffload', 'ldr', 'occ_p95', X_ldr_q, y_ldr_occ_p95, scale=False,
                          flavor='sqrtload', col_idx_arate=0, col_idx_meansvctime=18, load_pctile=0.95)

prob_blockedby_pp_q_erlangc_results = \
    crossval_summarize_mm('prob_blockedby_pp_q_erlangc', 'ldr', 'prob_blockedby_pp',
                          X_ldr_q, y_prob_blockedby_pp,
                          scale=False, fit_intercept=True,
                          flavor='erlangc', col_idx_arate=0, col_idx_meansvctime=4, col_idx_numservers=6)

condmeantime_blockedby_pp_q_mgc_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_q_mgc', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_q, y_condmeantime_blockedby_pp,
                          scale=False, fit_intercept=True,
                          flavor='condmeanwaitldr', col_idx_arate=0, col_idx_meansvctime=4, col_idx_numservers=6,
                          col_idx_cv2svctime=15)

# Linear models using only queueing approximation terms
ldr_occ_mean_onlyq_lm_results = \
    crossval_summarize_mm('ldr_occ_mean_onlyq_lm', 'ldr', 'occ_mean',
                          X_ldr_occ_mean_onlyq, y_ldr_occ_mean, scale=False, flavor='lm')

ldr_occ_p95_onlyq_lm_results = \
    crossval_summarize_mm('ldr_occ_p95_onlyq_lm', 'ldr', 'occ_p95',
                          X_ldr_occ_p95_onlyq, y_ldr_occ_p95, scale=False, flavor='lm')

prob_blockedby_pp_onlyq_lm_results = \
    crossval_summarize_mm('prob_blockedby_pp_onlyq_lm', 'ldr', 'pct_blockedby_pp',
                          X_ldr_prob_blockedby_pp_onlyq, y_prob_blockedby_pp, scale=False, flavor='lm')

condmeantime_blockedby_pp_onlyq_lm_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_onlyq_lm', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_condmeantime_blockedby_pp_onlyq, y_condmeantime_blockedby_pp, scale=False, flavor='lm')

## Linear regression (lm)
ldr_occ_mean_basicq_lm_results = \
    crossval_summarize_mm('ldr_occ_mean_basicq_lm', 'ldr', 'occ_mean',
                          X_ldr_basicq, y_ldr_occ_mean, scale=False, flavor='lm')

ldr_occ_mean_q_lm_results = \
    crossval_summarize_mm('ldr_occ_mean_q_lm', 'ldr', 'occ_mean',
                          X_ldr_q, y_ldr_occ_mean, scale=False, flavor='lm')

ldr_occ_mean_noq_lm_results = \
    crossval_summarize_mm('ldr_occ_mean_noq_lm', 'ldr', 'occ_mean',
                          X_ldr_noq, y_ldr_occ_mean, scale=False, flavor='lm')

ldr_occ_p95_basicq_lm_results = \
    crossval_summarize_mm('ldr_occ_p95_basicq_lm', 'ldr', 'occ_p95',
                          X_ldr_basicq, y_ldr_occ_p95, scale=False, flavor='lm')

ldr_occ_p95_q_lm_results = \
    crossval_summarize_mm('ldr_occ_p95_q_lm', 'ldr', 'occ_p95', X_ldr_q, y_ldr_occ_p95, scale=False, flavor='lm')

ldr_occ_p95_noq_lm_results = \
    crossval_summarize_mm('ldr_occ_p95_noq_lm', 'ldr', 'occ_p95', X_ldr_noq, y_ldr_occ_p95, scale=False, flavor='lm')

prob_blockedby_pp_basicq_lm_results = \
    crossval_summarize_mm('prob_blockedby_pp_basicq_lm', 'ldr', 'pct_blockedby_pp',
                          X_ldr_basicq, y_prob_blockedby_pp,
                          scale=False, fit_intercept=True, flavor='lm')

prob_blockedby_pp_q_lm_results = \
    crossval_summarize_mm('prob_blockedby_pp_q_lm', 'ldr', 'pct_blockedby_pp',
                          X_ldr_q, y_prob_blockedby_pp,
                          scale=False, fit_intercept=True, flavor='lm')

prob_blockedby_pp_noq_lm_results = \
    crossval_summarize_mm('prob_blockedby_pp_noq_lm', 'ldr', 'pct_blockedby_pp',
                          X_ldr_noq, y_prob_blockedby_pp,
                          scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedby_pp_basicq_lm_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_basicq_lm', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_basicq, y_condmeantime_blockedby_pp,
                          scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedby_pp_q_lm_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_q_lm', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_q, y_condmeantime_blockedby_pp,
                          scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedby_pp_noq_lm_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_noq_lm', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_noq, y_condmeantime_blockedby_pp,
                          scale=False, fit_intercept=True, flavor='lm')

# LassoCV (lassocv)
ldr_occ_mean_basicq_lassocv_results = \
    crossval_summarize_mm('ldr_occ_mean_basicq_lassocv', 'ldr', 'occ_mean', X_ldr_basicq, y_ldr_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

ldr_occ_mean_q_lassocv_results = \
    crossval_summarize_mm('ldr_occ_mean_q_lassocv', 'ldr', 'occ_mean', X_ldr_q, y_ldr_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

ldr_occ_mean_noq_lassocv_results = \
    crossval_summarize_mm('ldr_occ_mean_noq_lassocv', 'ldr', 'occ_mean', X_ldr_noq, y_ldr_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

ldr_occ_p95_basicq_lassocv_results = \
    crossval_summarize_mm('ldr_occ_p95_basicq_lassocv', 'ldr', 'occ_p95', X_ldr_basicq, y_ldr_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

ldr_occ_p95_q_lassocv_results = \
    crossval_summarize_mm('ldr_occ_p95_q_lassocv', 'ldr', 'occ_p95', X_ldr_q, y_ldr_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

ldr_occ_p95_noq_lassocv_results = \
    crossval_summarize_mm('ldr_occ_p95_noq_lassocv', 'ldr', 'occ_p95', X_ldr_noq, y_ldr_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

prob_blockedby_pp_basicq_lassocv_results = \
    crossval_summarize_mm('prob_blockedby_pp_basicq_lassocv', 'ldr', 'pct_blockedby_pp',
                          X_ldr_basicq, y_prob_blockedby_pp,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

prob_blockedby_pp_q_lassocv_results = \
    crossval_summarize_mm('prob_blockedby_pp_q_lassocv', 'ldr', 'pct_blockedby_pp',
                          X_ldr_q, y_prob_blockedby_pp,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

prob_blockedby_pp_noq_lassocv_results = \
    crossval_summarize_mm('prob_blockedby_pp_noq_lassocv', 'ldr', 'pct_blockedby_pp',
                          X_ldr_noq, y_prob_blockedby_pp,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

condmeantime_blockedby_pp_basicq_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_basicq_lassocv', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_basicq, y_condmeantime_blockedby_pp,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

condmeantime_blockedby_pp_q_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_q_lassocv', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_q, y_condmeantime_blockedby_pp,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

condmeantime_blockedby_pp_noq_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_noq_lassocv', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_noq, y_condmeantime_blockedby_pp,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

# Polynomial regression (poly)
ldr_occ_mean_basicq_poly_results = \
    crossval_summarize_mm('ldr_occ_mean_basicq_poly', 'ldr', 'occ_mean',
                          X_ldr_basicq, y_ldr_occ_mean, scale=False, flavor='poly')

ldr_occ_mean_q_poly_results = \
    crossval_summarize_mm('ldr_occ_mean_q_poly', 'ldr', 'occ_mean',
                          X_ldr_q, y_ldr_occ_mean, scale=False, flavor='poly')

ldr_occ_mean_noq_poly_results = \
    crossval_summarize_mm('ldr_occ_mean_noq_poly', 'ldr', 'occ_mean',
                          X_ldr_noq, y_ldr_occ_mean, scale=False,
                          flavor='poly')

ldr_occ_p95_basicq_poly_results = \
    crossval_summarize_mm('ldr_occ_p95_basicq_poly', 'ldr', 'occ_p95',
                          X_ldr_basicq, y_ldr_occ_p95, scale=False, flavor='poly')

ldr_occ_p95_q_poly_results = \
    crossval_summarize_mm('ldr_occ_p95_q_poly', 'ldr', 'occ_p95',
                          X_ldr_q, y_ldr_occ_p95, scale=False, flavor='poly')

ldr_occ_p95_noq_poly_results = \
    crossval_summarize_mm('ldr_occ_p95_noq_poly', 'ldr', 'occ_p95',
                          X_ldr_noq, y_ldr_occ_p95, scale=False, flavor='poly')

prob_blockedby_pp_basicq_poly_results = \
    crossval_summarize_mm('prob_blockedby_pp_basicq_poly', 'ldr', 'pct_blockedby_pp',
                          X_ldr_basicq, y_prob_blockedby_pp,
                          scale=False, flavor='poly')

prob_blockedby_pp_q_poly_results = \
    crossval_summarize_mm('prob_blockedby_pp_q_poly', 'ldr', 'pct_blockedby_pp',
                          X_ldr_q, y_prob_blockedby_pp,
                          scale=False, flavor='poly')

prob_blockedby_pp_noq_poly_results = \
    crossval_summarize_mm('prob_blockedby_pp_noq_poly', 'ldr', 'pct_blockedby_pp',
                          X_ldr_noq, y_prob_blockedby_pp,
                          scale=False, flavor='poly')

condmeantime_blockedby_pp_basicq_poly_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_basicq_poly', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_basicq, y_condmeantime_blockedby_pp,
                          scale=False, flavor='poly')

condmeantime_blockedby_pp_q_poly_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_q_poly', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_q, y_condmeantime_blockedby_pp,
                          scale=False, flavor='poly')

condmeantime_blockedby_pp_noq_poly_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_noq_poly', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_noq, y_condmeantime_blockedby_pp,
                          scale=False, flavor='poly')

# Random forest (rf)
ldr_occ_mean_basicq_rf_results = \
    crossval_summarize_mm('ldr_occ_mean_basicq_rf', 'ldr', 'occ_mean',
                          X_ldr_basicq, y_ldr_occ_mean, scale=False, flavor='rf')

ldr_occ_mean_q_rf_results = \
    crossval_summarize_mm('ldr_occ_mean_q_rf', 'ldr', 'occ_mean',
                          X_ldr_q, y_ldr_occ_mean, scale=False, flavor='rf')

ldr_occ_mean_noq_rf_results = \
    crossval_summarize_mm('ldr_occ_mean_noq_rf', 'ldr', 'occ_mean',
                          X_ldr_noq, y_ldr_occ_mean, scale=False, flavor='rf')

ldr_occ_p95_basicq_rf_results = \
    crossval_summarize_mm('ldr_occ_p95_basicq_rf', 'ldr', 'occ_p95',
                          X_ldr_basicq, y_ldr_occ_p95, scale=False, flavor='rf')

ldr_occ_p95_q_rf_results = \
    crossval_summarize_mm('ldr_occ_p95_q_rf', 'ldr', 'occ_p95',
                          X_ldr_q, y_ldr_occ_p95, scale=False, flavor='rf')

ldr_occ_p95_noq_rf_results = \
    crossval_summarize_mm('ldr_occ_p95_noq_rf', 'ldr', 'occ_p95',
                          X_ldr_noq, y_ldr_occ_p95, scale=False, flavor='rf')

prob_blockedby_pp_basicq_rf_results = \
    crossval_summarize_mm('prob_blockedby_pp_q_rf', 'ldr', 'pct_blockedby_pp',
                          X_ldr_basicq, y_prob_blockedby_pp,
                          scale=False, flavor='rf')

prob_blockedby_pp_q_rf_results = \
    crossval_summarize_mm('prob_blockedby_pp_q_rf', 'ldr', 'pct_blockedby_pp',
                          X_ldr_q, y_prob_blockedby_pp,
                          scale=False, flavor='rf')

prob_blockedby_pp_noq_rf_results = \
    crossval_summarize_mm('prob_blockedby_pp_noq_rf', 'ldr', 'pct_blockedby_pp',
                          X_ldr_noq, y_prob_blockedby_pp,
                          scale=False, flavor='rf')

condmeantime_blockedby_pp_basicq_rf_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_basicq_rf', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_basicq, y_condmeantime_blockedby_pp,
                          scale=False, flavor='rf')

condmeantime_blockedby_pp_q_rf_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_q_rf', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_q, y_condmeantime_blockedby_pp,
                          scale=False, flavor='rf')

condmeantime_blockedby_pp_noq_rf_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_noq_rf', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_noq, y_condmeantime_blockedby_pp,
                          scale=False, flavor='rf')

# Support vector regression (svr)
ldr_occ_mean_basicq_svr_results = \
    crossval_summarize_mm('ldr_occ_mean_basicq_svr', 'ldr', 'occ_mean',
                          X_ldr_basicq, y_ldr_occ_mean, flavor='svr', scale=True)

ldr_occ_mean_q_svr_results = \
    crossval_summarize_mm('ldr_occ_mean_q_svr', 'ldr', 'occ_mean',
                          X_ldr_q, y_ldr_occ_mean, flavor='svr', scale=True)

ldr_occ_mean_noq_svr_results = \
    crossval_summarize_mm('ldr_occ_mean_noq_svr', 'ldr', 'occ_mean',
                          X_ldr_noq, y_ldr_occ_mean, flavor='svr', scale=True)

ldr_occ_p95_basicq_svr_results = \
    crossval_summarize_mm('ldr_occ_p95_basicq_svr', 'ldr', 'occ_p95',
                          X_ldr_basicq, y_ldr_occ_p95, flavor='svr', scale=True)

ldr_occ_p95_q_svr_results = \
    crossval_summarize_mm('ldr_occ_p95_q_svr', 'ldr', 'occ_p95',
                          X_ldr_q, y_ldr_occ_p95, flavor='svr', scale=True)

ldr_occ_p95_noq_svr_results = \
    crossval_summarize_mm('ldr_occ_p95_noq_svr', 'ldr', 'occ_p95',
                          X_ldr_noq, y_ldr_occ_p95, flavor='svr', scale=True)

prob_blockedby_pp_basicq_svr_results = \
    crossval_summarize_mm('prob_blockedby_pp_basicq_svr', 'ldr', 'pct_blockedby_pp',
                          X_ldr_basicq, y_prob_blockedby_pp,
                          scale=True, flavor='svr')

prob_blockedby_pp_q_svr_results = \
    crossval_summarize_mm('prob_blockedby_pp_q_svr', 'ldr', 'pct_blockedby_pp',
                          X_ldr_q, y_prob_blockedby_pp,
                          scale=True, flavor='svr')

prob_blockedby_pp_noq_svr_results = \
    crossval_summarize_mm('prob_blockedby_pp_noq_svr', 'ldr', 'pct_blockedby_pp',
                          X_ldr_noq, y_prob_blockedby_pp,
                          scale=True, flavor='svr')

condmeantime_blockedby_pp_basicq_svr_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_basicq_svr', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_basicq, y_condmeantime_blockedby_pp,
                          scale=True, flavor='svr')

condmeantime_blockedby_pp_q_svr_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_q_svr', 'ldr', 'condmeantime_blockedbyldr',
                          X_ldr_q, y_condmeantime_blockedby_pp,
                          scale=True, flavor='svr')

condmeantime_blockedby_pp_noq_svr_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_noq_svr', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_noq, y_condmeantime_blockedby_pp,
                          scale=True, flavor='svr')

# MLPRegressor Neural net (nn)
ldr_occ_mean_basicq_nn_results = \
    crossval_summarize_mm('ldr_occ_mean_basicq_nn', 'ldr', 'occ_mean',
                          X_ldr_basicq, y_ldr_occ_mean, flavor='nn', scale=True)

ldr_occ_mean_q_nn_results = \
    crossval_summarize_mm('ldr_occ_mean_q_nn', 'ldr', 'occ_mean',
                          X_ldr_q, y_ldr_occ_mean, flavor='nn', scale=True)

ldr_occ_mean_noq_nn_results = \
    crossval_summarize_mm('ldr_occ_mean_noq_nn', 'ldr', 'occ_mean', X_ldr_noq, y_ldr_occ_mean, flavor='nn', scale=True)

ldr_occ_p95_basicq_nn_results = \
    crossval_summarize_mm('ldr_occ_p95_basicq_nn', 'ldr', 'occ_p95',
                          X_ldr_basicq, y_ldr_occ_p95, flavor='nn', scale=True)

ldr_occ_p95_q_nn_results = \
    crossval_summarize_mm('ldr_occ_p95_q_nn', 'ldr', 'occ_p95',
                          X_ldr_q, y_ldr_occ_p95, flavor='nn', scale=True)

ldr_occ_p95_noq_nn_results = \
    crossval_summarize_mm('ldr_occ_p95_noq_nn', 'ldr', 'occ_p95',
                          X_ldr_noq, y_ldr_occ_p95, flavor='nn', scale=True)

prob_blockedby_pp_basicq_nn_results = \
    crossval_summarize_mm('prob_blockedby_pp_basicq_nn', 'ldr', 'pct_blockedby_pp',
                          X_ldr_basicq, y_prob_blockedby_pp,
                          scale=True, flavor='nn')

prob_blockedby_pp_q_nn_results = \
    crossval_summarize_mm('prob_blockedby_pp_q_nn', 'ldr', 'pct_blockedby_pp',
                          X_ldr_q, y_prob_blockedby_pp,
                          scale=True, flavor='nn')

prob_blockedby_pp_noq_nn_results = \
    crossval_summarize_mm('prob_blockedby_pp_noq_nn', 'ldr', 'pct_blockedby_pp',
                          X_ldr_noq, y_prob_blockedby_pp,
                          scale=True, flavor='nn')

condmeantime_blockedby_pp_basicq_nn_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_basicq_nn', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_basicq, y_condmeantime_blockedby_pp,
                          scale=True, flavor='nn')

condmeantime_blockedby_pp_q_nn_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_q_nn', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_q, y_condmeantime_blockedby_pp,
                          scale=True, flavor='nn')

condmeantime_blockedby_pp_noq_nn_results = \
    crossval_summarize_mm('condmeantime_blockedby_pp_noq_nn', 'ldr', 'condmeantime_blockedby_pp',
                          X_ldr_noq, y_condmeantime_blockedby_pp,
                          scale=True, flavor='nn')




ldr_results = {'ldr_occ_mean_basicq_lm_results': ldr_occ_mean_basicq_lm_results,
               'ldr_occ_mean_q_lm_results': ldr_occ_mean_q_lm_results,
               'ldr_occ_mean_noq_lm_results': ldr_occ_mean_noq_lm_results,
               'ldr_occ_mean_basicq_lassocv_results': ldr_occ_mean_basicq_lassocv_results,
               'ldr_occ_mean_q_lassocv_results': ldr_occ_mean_q_lassocv_results,
               'ldr_occ_mean_noq_lassocv_results': ldr_occ_mean_noq_lassocv_results,
               'ldr_occ_mean_basicq_poly_results': ldr_occ_mean_basicq_poly_results,
               'ldr_occ_mean_q_poly_results': ldr_occ_mean_q_poly_results,
               'ldr_occ_mean_noq_poly_results': ldr_occ_mean_noq_poly_results,
               'ldr_occ_mean_basicq_rf_results': ldr_occ_mean_basicq_rf_results,
               'ldr_occ_mean_q_rf_results': ldr_occ_mean_q_rf_results,
               'ldr_occ_mean_noq_rf_results': ldr_occ_mean_noq_rf_results,
               'ldr_occ_mean_basicq_svr_results': ldr_occ_mean_basicq_svr_results,
               'ldr_occ_mean_q_svr_results': ldr_occ_mean_q_svr_results,
               'ldr_occ_mean_noq_svr_results': ldr_occ_mean_noq_svr_results,
               'ldr_occ_mean_basicq_nn_results': ldr_occ_mean_basicq_nn_results,
               'ldr_occ_mean_q_nn_results': ldr_occ_mean_q_nn_results,
               'ldr_occ_mean_noq_nn_results': ldr_occ_mean_noq_nn_results,
               'ldr_occ_mean_q_load_results': ldr_occ_mean_q_load_results,
               'ldr_occ_p95_basicq_lm_results': ldr_occ_p95_basicq_lm_results,
               'ldr_occ_p95_q_lm_results': ldr_occ_p95_q_lm_results,
               'ldr_occ_p95_noq_lm_results': ldr_occ_p95_noq_lm_results,
               'ldr_occ_p95_basicq_lassocv_results': ldr_occ_p95_basicq_lassocv_results,
               'ldr_occ_p95_q_lassocv_results': ldr_occ_p95_q_lassocv_results,
               'ldr_occ_p95_noq_lassocv_results': ldr_occ_p95_noq_lassocv_results,
               'ldr_occ_p95_basicq_poly_results': ldr_occ_p95_basicq_poly_results,
               'ldr_occ_p95_q_poly_results': ldr_occ_p95_q_poly_results,
               'ldr_occ_p95_noq_poly_results': ldr_occ_p95_noq_poly_results,
               'ldr_occ_p95_basicq_rf_results': ldr_occ_p95_basicq_rf_results,
               'ldr_occ_p95_q_rf_results': ldr_occ_p95_q_rf_results,
               'ldr_occ_p95_noq_rf_results': ldr_occ_p95_noq_rf_results,
               'ldr_occ_p95_basicq_svr_results': ldr_occ_p95_basicq_svr_results,
               'ldr_occ_p95_q_svr_results': ldr_occ_p95_q_svr_results,
               'ldr_occ_p95_noq_svr_results': ldr_occ_p95_noq_svr_results,
               'ldr_occ_p95_basicq_nn_results': ldr_occ_p95_basicq_nn_results,
               'ldr_occ_p95_q_nn_results': ldr_occ_p95_q_nn_results,
               'ldr_occ_p95_noq_nn_results': ldr_occ_p95_noq_nn_results,
               'ldr_occ_p95_q_sqrtload_results': ldr_occ_p95_q_sqrtload_results,
               'prob_blockedby_pp_basicq_lm_results': prob_blockedby_pp_basicq_lm_results,
               'prob_blockedby_pp_q_lm_results': prob_blockedby_pp_q_lm_results,
               'prob_blockedby_pp_noq_lm_results': prob_blockedby_pp_noq_lm_results,
               'prob_blockedby_pp_basicq_lassocv_results': prob_blockedby_pp_basicq_lassocv_results,
               'prob_blockedby_pp_q_lassocv_results': prob_blockedby_pp_q_lassocv_results,
               'prob_blockedby_pp_noq_lassocv_results': prob_blockedby_pp_noq_lassocv_results,
               'prob_blockedby_pp_basicq_poly_results': prob_blockedby_pp_basicq_poly_results,
               'prob_blockedby_pp_q_poly_results': prob_blockedby_pp_q_poly_results,
               'prob_blockedby_pp_noq_poly_results': prob_blockedby_pp_noq_poly_results,
               'prob_blockedby_pp_basicq_rf_results': prob_blockedby_pp_basicq_rf_results,
               'prob_blockedby_pp_q_rf_results': prob_blockedby_pp_q_rf_results,
               'prob_blockedby_pp_noq_rf_results': prob_blockedby_pp_noq_rf_results,
               'prob_blockedby_pp_basicq_svr_results': prob_blockedby_pp_basicq_svr_results,
               'prob_blockedby_pp_q_svr_results': prob_blockedby_pp_q_svr_results,
               'prob_blockedby_pp_noq_svr_results': prob_blockedby_pp_noq_svr_results,
               'prob_blockedby_pp_basicq_nn_results': prob_blockedby_pp_basicq_nn_results,
               'prob_blockedby_pp_q_nn_results': prob_blockedby_pp_q_nn_results,
               'prob_blockedby_pp_noq_nn_results': prob_blockedby_pp_noq_nn_results,
               'prob_blockedby_pp_q_erlangc_results': prob_blockedby_pp_q_erlangc_results,
               'condmeantime_blockedby_pp_basicq_lm_results': condmeantime_blockedby_pp_basicq_lm_results,
               'condmeantime_blockedby_pp_q_lm_results': condmeantime_blockedby_pp_q_lm_results,
               'condmeantime_blockedby_pp_noq_lm_results': condmeantime_blockedby_pp_noq_lm_results,
               'condmeantime_blockedby_pp_basicq_lassocv_results': condmeantime_blockedby_pp_basicq_lassocv_results,
               'condmeantime_blockedby_pp_q_lassocv_results': condmeantime_blockedby_pp_q_lassocv_results,
               'condmeantime_blockedby_pp_noq_lassocv_results': condmeantime_blockedby_pp_noq_lassocv_results,
               'condmeantime_blockedby_pp_basicq_poly_results': condmeantime_blockedby_pp_basicq_poly_results,
               'condmeantime_blockedby_pp_q_poly_results': condmeantime_blockedby_pp_q_poly_results,
               'condmeantime_blockedby_pp_noq_poly_results': condmeantime_blockedby_pp_noq_poly_results,
               'condmeantime_blockedby_pp_basicq_rf_results': condmeantime_blockedby_pp_basicq_rf_results,
               'condmeantime_blockedby_pp_q_rf_results': condmeantime_blockedby_pp_q_rf_results,
               'condmeantime_blockedby_pp_noq_rf_results': condmeantime_blockedby_pp_noq_rf_results,
               'condmeantime_blockedby_pp_basicq_svr_results': condmeantime_blockedby_pp_basicq_svr_results,
               'condmeantime_blockedby_pp_q_svr_results': condmeantime_blockedby_pp_q_svr_results,
               'condmeantime_blockedby_pp_noq_svr_results': condmeantime_blockedby_pp_noq_svr_results,
               'condmeantime_blockedby_pp_basicq_nn_results': condmeantime_blockedby_pp_basicq_nn_results,
               'condmeantime_blockedby_pp_q_nn_results': condmeantime_blockedby_pp_q_nn_results,
               'condmeantime_blockedby_pp_noq_nn_results': condmeantime_blockedby_pp_noq_nn_results,
               'condmeantime_blockedby_pp_q_mgc_results' : condmeantime_blockedby_pp_q_mgc_results,
               'ldr_occ_mean_onlyq_lm_results': ldr_occ_mean_onlyq_lm_results,
               'ldr_occ_p95_onlyq_lm_results': ldr_occ_p95_onlyq_lm_results,
               'prob_blockedby_pp_onlyq_lm_results': prob_blockedby_pp_onlyq_lm_results,
               'condmeantime_blockedby_pp_onlyq_lm_results': condmeantime_blockedby_pp_onlyq_lm_results
               }

# Pickle the results
with open(Path(output_path, pickle_filename), 'wb') as pickle_file:
    pickle.dump(ldr_results, pickle_file)
