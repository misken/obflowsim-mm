from pathlib import Path
import pickle

import pandas as pd

from mmfitting import crossval_summarize_mm

experiment = "exp11"
data_path = Path("data")
output_path = Path("output")
figures_path = Path("output", "figures")
raw_data_path = Path("data", "raw")
pickle_filename = f"pp_results_{experiment}.pkl"

# X matrices
X_pp_noq = pd.read_csv(Path(data_path, f'X_pp_noq_{experiment}.csv'), index_col=0)
X_pp_basicq = pd.read_csv(Path(data_path, f'X_pp_basicq_{experiment}.csv'), index_col=0)

X_pp_occ_mean_onlyq = pd.read_csv(Path(data_path, f'X_pp_occmean_onlyq_{experiment}.csv'), index_col=0)
X_pp_occ_p95_onlyq = pd.read_csv(Path(data_path, f'X_pp_occp95_onlyq_{experiment}.csv'), index_col=0)

# y vectors
y_pp_occ_mean = pd.read_csv(Path(data_path, f'y_pp_occ_mean_{experiment}.csv'), index_col=0, squeeze=True)
y_pp_occ_p95 = pd.read_csv(Path(data_path, f'y_pp_occ_p95_{experiment}.csv'), index_col=0, squeeze=True)

# Fit models

# Linear regression (lm)
pp_occ_mean_basicq_lm_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_lm', 'pp', 'occ_mean', X_pp_basicq, y_pp_occ_mean, scale=False, flavor='lm')

pp_occ_mean_noq_lm_results = \
    crossval_summarize_mm('pp_occ_mean_noq_lm', 'pp', 'occ_mean', X_pp_noq, y_pp_occ_mean, scale=False, flavor='lm')


pp_occ_p95_basicq_lm_results = \
    crossval_summarize_mm('pp_occ_p95_basicq_lm', 'pp', 'occ_p95', X_pp_basicq, y_pp_occ_p95, scale=False, flavor='lm')

pp_occ_p95_noq_lm_results = \
    crossval_summarize_mm('pp_occ_p95_noq_lm', 'pp', 'occ_p95', X_pp_noq, y_pp_occ_p95, scale=False, flavor='lm')

# LassoCV (lassocv)
pp_occ_mean_basicq_lassocv_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_lassocv', 'pp', 'occ_mean', X_pp_basicq, y_pp_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

pp_occ_mean_noq_lassocv_results = \
    crossval_summarize_mm('pp_occ_mean_noq_lassocv', 'pp', 'occ_mean', X_pp_noq, y_pp_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

pp_occ_p95_basicq_lassocv_results = \
    crossval_summarize_mm('pp_occ_p95_basicq_lassocv', 'pp', 'occ_p95', X_pp_basicq, y_pp_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

pp_occ_p95_noq_lassocv_results = \
    crossval_summarize_mm('pp_occ_p95_noq_lassocv', 'pp', 'occ_p95', X_pp_noq, y_pp_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

# Polynomial regression (poly)
pp_occ_mean_basicq_poly_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_poly', 'pp', 'occ_mean', X_pp_basicq, y_pp_occ_mean, scale=False, flavor='poly')

pp_occ_mean_noq_poly_results = \
    crossval_summarize_mm('pp_occ_mean_noq_poly', 'pp', 'occ_mean', X_pp_noq, y_pp_occ_mean, scale=False, flavor='poly')


pp_occ_p95_basicq_poly_results = \
    crossval_summarize_mm('pp_occ_p95_basicq_poly', 'pp', 'occ_p95', X_pp_basicq, y_pp_occ_p95, scale=False, flavor='poly')

pp_occ_p95_noq_poly_results = \
    crossval_summarize_mm('pp_occ_p95_noq_poly', 'pp', 'occ_p95', X_pp_noq, y_pp_occ_p95, scale=False, flavor='poly')

# Random forest (rf)
pp_occ_mean_basicq_rf_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_rf', 'pp', 'occ_mean', X_pp_basicq, y_pp_occ_mean, scale=False, flavor='rf')

pp_occ_mean_noq_rf_results = \
    crossval_summarize_mm('pp_occ_mean_noq_rf', 'pp', 'occ_mean', X_pp_noq, y_pp_occ_mean, scale=False, flavor='rf')

pp_occ_p95_basicq_rf_results = \
    crossval_summarize_mm('pp_occ_p95_basicq_rf', 'pp', 'occ_p95', X_pp_basicq, y_pp_occ_p95, scale=False, flavor='rf')

pp_occ_p95_noq_rf_results = \
    crossval_summarize_mm('pp_occ_p95_noq_rf', 'pp', 'occ_p95', X_pp_noq, y_pp_occ_p95, scale=False, flavor='rf')

# Support vector regression (svr)
pp_occ_mean_basicq_svr_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_svr', 'pp', 'occ_mean', X_pp_basicq, y_pp_occ_mean, flavor='svr', scale=True)

pp_occ_mean_noq_svr_results = \
    crossval_summarize_mm('pp_occ_mean_noq_svr', 'pp', 'occ_mean', X_pp_noq, y_pp_occ_mean, flavor='svr', scale=True)


pp_occ_p95_basicq_svr_results = \
    crossval_summarize_mm('pp_occ_p95_basicq_svr', 'pp', 'occ_p95', X_pp_basicq, y_pp_occ_p95, flavor='svr', scale=True)

pp_occ_p95_noq_svr_results = \
    crossval_summarize_mm('pp_occ_p95_noq_svr', 'pp', 'occ_p95', X_pp_noq, y_pp_occ_p95, flavor='svr', scale=True)

# MLPRegressor Neural net (nn)
pp_occ_mean_basicq_nn_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_nn', 'pp', 'occ_mean', X_pp_basicq, y_pp_occ_mean, flavor='nn', scale=True)

pp_occ_mean_noq_nn_results = \
    crossval_summarize_mm('pp_occ_mean_noq_nn', 'pp', 'occ_mean', X_pp_noq, y_pp_occ_mean, flavor='nn', scale=True)

pp_occ_p95_basicq_nn_results = \
    crossval_summarize_mm('pp_occ_p95_basicq_nn', 'pp', 'occ_p95', X_pp_basicq, y_pp_occ_p95, flavor='nn', scale=True)

pp_occ_p95_noq_nn_results = \
    crossval_summarize_mm('pp_occ_p95_noq_nn', 'pp', 'occ_p95', X_pp_noq, y_pp_occ_p95, flavor='nn', scale=True)

# Load based models

pp_occ_mean_basicq_load_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_load', 'pp', 'occ_mean', X_pp_basicq, y_pp_occ_mean, scale=False,
                          flavor='load', col_idx_arate=0, col_idx_meansvctime=1)

pp_occ_p95_basicq_sqrtload_results = \
    crossval_summarize_mm('pp_occ_p95_basicq_sqrtload', 'pp', 'occ_p95', X_pp_basicq, y_pp_occ_p95, scale=False,
                          flavor='sqrtload', col_idx_arate=0, col_idx_meansvctime=1, load_pctile=0.95)

# Linear models using only queueing approximation terms
pp_occ_mean_onlyq_lm_results = \
    crossval_summarize_mm('pp_occ_mean_onlyq_lm', 'pp', 'occ_mean',
                          X_pp_occ_mean_onlyq, y_pp_occ_mean, scale=False, flavor='lm')

pp_occ_p95_onlyq_lm_results = \
    crossval_summarize_mm('pp_occ_p95_onlyq_lm', 'pp', 'occ_p95',
                          X_pp_occ_p95_onlyq, y_pp_occ_p95, scale=False, flavor='lm')

# HGBR
pp_occ_mean_basicq_hgbr_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_hgbr', 'pp', 'occ_mean', X_pp_basicq, y_pp_occ_mean, scale=False,
                          flavor='hgbr')

pp_occ_p95_basicq_hgbr_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_hgbr', 'pp', 'occ_mean', X_pp_basicq, y_pp_occ_p95, scale=False,
                          flavor='hgbr')

# Gather results

pp_results = {'pp_occ_mean_basicq_lm_results': pp_occ_mean_basicq_lm_results,
              'pp_occ_mean_noq_lm_results': pp_occ_mean_noq_lm_results,
              'pp_occ_mean_basicq_lassocv_results': pp_occ_mean_basicq_lassocv_results,
              'pp_occ_mean_noq_lassocv_results': pp_occ_mean_noq_lassocv_results,
              'pp_occ_mean_basicq_poly_results': pp_occ_mean_basicq_poly_results,
              'pp_occ_mean_noq_poly_results': pp_occ_mean_noq_poly_results,
              'pp_occ_mean_basicq_rf_results': pp_occ_mean_basicq_rf_results,
              'pp_occ_mean_noq_rf_results': pp_occ_mean_noq_rf_results,
              'pp_occ_mean_basicq_svr_results': pp_occ_mean_basicq_svr_results,
              'pp_occ_mean_noq_svr_results': pp_occ_mean_noq_svr_results,
              'pp_occ_mean_basicq_nn_results': pp_occ_mean_basicq_nn_results,
              'pp_occ_mean_noq_nn_results': pp_occ_mean_noq_nn_results,
              'pp_occ_mean_onlyq_lm_results': pp_occ_mean_onlyq_lm_results,
              'pp_occ_p95_basicq_lm_results': pp_occ_p95_basicq_lm_results,
              'pp_occ_p95_noq_lm_results': pp_occ_p95_noq_lm_results,
              'pp_occ_p95_basicq_lassocv_results': pp_occ_p95_basicq_lassocv_results,
              'pp_occ_p95_noq_lassocv_results': pp_occ_p95_noq_lassocv_results,
              'pp_occ_p95_basicq_poly_results': pp_occ_p95_basicq_poly_results,
              'pp_occ_p95_noq_poly_results': pp_occ_p95_noq_poly_results,
              'pp_occ_p95_basicq_rf_results': pp_occ_p95_basicq_rf_results,
              'pp_occ_p95_noq_rf_results': pp_occ_p95_noq_rf_results,
              'pp_occ_p95_basicq_svr_results': pp_occ_p95_basicq_svr_results,
              'pp_occ_p95_noq_svr_results': pp_occ_p95_noq_svr_results,
              'pp_occ_p95_basicq_nn_results': pp_occ_p95_basicq_nn_results,
              'pp_occ_p95_noq_nn_results': pp_occ_p95_noq_nn_results,
              'pp_occ_mean_basicq_load_results': pp_occ_mean_basicq_load_results,
              'pp_occ_p95_basicq_sqrtload_results': pp_occ_p95_basicq_sqrtload_results,
              'pp_occ_mean_basicq_hgbr_results': pp_occ_mean_basicq_hgbr_results,
              'pp_occ_p95_basicq_hgbr_results': pp_occ_p95_basicq_hgbr_results,
              'pp_occ_p95_onlyq_lm_results': pp_occ_p95_onlyq_lm_results
              }


# Pickle the results
with open(Path(output_path, pickle_filename), 'wb') as pickle_file:
    pickle.dump(pp_results, pickle_file)