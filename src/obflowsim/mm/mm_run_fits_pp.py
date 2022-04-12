import sys
import argparse
from pathlib import Path
import pickle

import pandas as pd

from obflowsim.mm.mm_fitting import crossval_summarize_mm
from obflowsim.mm.mm_process_fitted_models import create_cv_plots, create_coeff_plots
from obflowsim.mm.mm_process_fitted_models import create_metrics_df, create_predictions_df

UNIT = "pp"


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
    X_pp_noq = pd.read_csv(Path(input_path, f'X_pp_noq_{experiment}.csv'), index_col=0)
    X_pp_basicq = pd.read_csv(Path(input_path, f'X_pp_basicq_{experiment}.csv'), index_col=0)

    X_pp_occmean_onlyq = pd.read_csv(Path(input_path, f'X_pp_occmean_onlyq_{experiment}.csv'), index_col=0)
    X_pp_occp95_onlyq = pd.read_csv(Path(input_path, f'X_pp_occp95_onlyq_{experiment}.csv'), index_col=0)

    # y vectors
    y_pp_occmean = pd.read_csv(Path(input_path, f'y_pp_occmean_{experiment}.csv'), index_col=0).squeeze("columns")
    y_pp_occp95 = pd.read_csv(Path(input_path, f'y_pp_occp95_{experiment}.csv'), index_col=0).squeeze("columns")

    # Fit models

    # Linear regression (lm)
    pp_occmean_basicq_lm_results = \
        crossval_summarize_mm('pp_occmean_basicq_lm', 'pp', 'occmean', X_pp_basicq, y_pp_occmean, scale=False, flavor='lm')

    pp_occmean_noq_lm_results = \
        crossval_summarize_mm('pp_occmean_noq_lm', 'pp', 'occmean', X_pp_noq, y_pp_occmean, scale=False, flavor='lm')


    pp_occp95_basicq_lm_results = \
        crossval_summarize_mm('pp_occp95_basicq_lm', 'pp', 'occp95', X_pp_basicq, y_pp_occp95, scale=False, flavor='lm')

    pp_occp95_noq_lm_results = \
        crossval_summarize_mm('pp_occp95_noq_lm', 'pp', 'occp95', X_pp_noq, y_pp_occp95, scale=False, flavor='lm')

    # LassoCV (lassocv)
    pp_occmean_basicq_lassocv_results = \
        crossval_summarize_mm('pp_occmean_basicq_lassocv', 'pp', 'occmean', X_pp_basicq, y_pp_occmean,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    pp_occmean_noq_lassocv_results = \
        crossval_summarize_mm('pp_occmean_noq_lassocv', 'pp', 'occmean', X_pp_noq, y_pp_occmean,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    pp_occp95_basicq_lassocv_results = \
        crossval_summarize_mm('pp_occp95_basicq_lassocv', 'pp', 'occp95', X_pp_basicq, y_pp_occp95,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    pp_occp95_noq_lassocv_results = \
        crossval_summarize_mm('pp_occp95_noq_lassocv', 'pp', 'occp95', X_pp_noq, y_pp_occp95,
                              scale=True, flavor='lassocv', lasso_max_iter=3000)

    # Polynomial regression (poly)
    pp_occmean_basicq_poly_results = \
        crossval_summarize_mm('pp_occmean_basicq_poly', 'pp', 'occmean', X_pp_basicq, y_pp_occmean, scale=False, flavor='poly')

    pp_occmean_noq_poly_results = \
        crossval_summarize_mm('pp_occmean_noq_poly', 'pp', 'occmean', X_pp_noq, y_pp_occmean, scale=False, flavor='poly')


    pp_occp95_basicq_poly_results = \
        crossval_summarize_mm('pp_occp95_basicq_poly', 'pp', 'occp95', X_pp_basicq, y_pp_occp95, scale=False, flavor='poly')

    pp_occp95_noq_poly_results = \
        crossval_summarize_mm('pp_occp95_noq_poly', 'pp', 'occp95', X_pp_noq, y_pp_occp95, scale=False, flavor='poly')

    # Random forest (rf)
    pp_occmean_basicq_rf_results = \
        crossval_summarize_mm('pp_occmean_basicq_rf', 'pp', 'occmean', X_pp_basicq, y_pp_occmean, scale=False, flavor='rf')

    pp_occmean_noq_rf_results = \
        crossval_summarize_mm('pp_occmean_noq_rf', 'pp', 'occmean', X_pp_noq, y_pp_occmean, scale=False, flavor='rf')

    pp_occp95_basicq_rf_results = \
        crossval_summarize_mm('pp_occp95_basicq_rf', 'pp', 'occp95', X_pp_basicq, y_pp_occp95, scale=False, flavor='rf')

    pp_occp95_noq_rf_results = \
        crossval_summarize_mm('pp_occp95_noq_rf', 'pp', 'occp95', X_pp_noq, y_pp_occp95, scale=False, flavor='rf')

    # Support vector regression (svr)
    pp_occmean_basicq_svr_results = \
        crossval_summarize_mm('pp_occmean_basicq_svr', 'pp', 'occmean', X_pp_basicq, y_pp_occmean, flavor='svr', scale=True)

    pp_occmean_noq_svr_results = \
        crossval_summarize_mm('pp_occmean_noq_svr', 'pp', 'occmean', X_pp_noq, y_pp_occmean, flavor='svr', scale=True)


    pp_occp95_basicq_svr_results = \
        crossval_summarize_mm('pp_occp95_basicq_svr', 'pp', 'occp95', X_pp_basicq, y_pp_occp95, flavor='svr', scale=True)

    pp_occp95_noq_svr_results = \
        crossval_summarize_mm('pp_occp95_noq_svr', 'pp', 'occp95', X_pp_noq, y_pp_occp95, flavor='svr', scale=True)

    # MLPRegressor Neural net (nn)
    pp_occmean_basicq_nn_results = \
        crossval_summarize_mm('pp_occmean_basicq_nn', 'pp', 'occmean', X_pp_basicq, y_pp_occmean, flavor='nn', scale=True)

    pp_occmean_noq_nn_results = \
        crossval_summarize_mm('pp_occmean_noq_nn', 'pp', 'occmean', X_pp_noq, y_pp_occmean, flavor='nn', scale=True)

    pp_occp95_basicq_nn_results = \
        crossval_summarize_mm('pp_occp95_basicq_nn', 'pp', 'occp95', X_pp_basicq, y_pp_occp95, flavor='nn', scale=True)

    pp_occp95_noq_nn_results = \
        crossval_summarize_mm('pp_occp95_noq_nn', 'pp', 'occp95', X_pp_noq, y_pp_occp95, flavor='nn', scale=True)

    # Load based models

    pp_occmean_basicq_load_results = \
        crossval_summarize_mm('pp_occmean_basicq_load', 'pp', 'occmean', X_pp_basicq, y_pp_occmean, scale=False,
                              flavor='load', col_idx_arate=0, col_idx_meansvctime=1)

    pp_occp95_basicq_sqrtload_results = \
        crossval_summarize_mm('pp_occp95_basicq_sqrtload', 'pp', 'occp95', X_pp_basicq, y_pp_occp95, scale=False,
                              flavor='sqrtload', col_idx_arate=0, col_idx_meansvctime=1, load_pctile=0.95)

    # Linear models using only queueing approximation terms
    pp_occmean_onlyq_lm_results = \
        crossval_summarize_mm('pp_occmean_onlyq_lm', 'pp', 'occmean',
                              X_pp_occmean_onlyq, y_pp_occmean, scale=False, flavor='lm')

    pp_occp95_onlyq_lm_results = \
        crossval_summarize_mm('pp_occp95_onlyq_lm', 'pp', 'occp95',
                              X_pp_occp95_onlyq, y_pp_occp95, scale=False, flavor='lm')

    # HGBR
    pp_occmean_basicq_hgbr_results = \
        crossval_summarize_mm('pp_occmean_basicq_hgbr', 'pp', 'occmean', X_pp_basicq, y_pp_occmean, scale=False,
                              flavor='hgbr')

    pp_occp95_basicq_hgbr_results = \
        crossval_summarize_mm('pp_occmean_basicq_hgbr', 'pp', 'occmean', X_pp_basicq, y_pp_occp95, scale=False,
                              flavor='hgbr')

    # Gather results

    pp_results = {'pp_occmean_basicq_lm_results': pp_occmean_basicq_lm_results,
                  'pp_occmean_noq_lm_results': pp_occmean_noq_lm_results,
                  'pp_occmean_basicq_lassocv_results': pp_occmean_basicq_lassocv_results,
                  'pp_occmean_noq_lassocv_results': pp_occmean_noq_lassocv_results,
                  'pp_occmean_basicq_poly_results': pp_occmean_basicq_poly_results,
                  'pp_occmean_noq_poly_results': pp_occmean_noq_poly_results,
                  'pp_occmean_basicq_rf_results': pp_occmean_basicq_rf_results,
                  'pp_occmean_noq_rf_results': pp_occmean_noq_rf_results,
                  'pp_occmean_basicq_svr_results': pp_occmean_basicq_svr_results,
                  'pp_occmean_noq_svr_results': pp_occmean_noq_svr_results,
                  'pp_occmean_basicq_nn_results': pp_occmean_basicq_nn_results,
                  'pp_occmean_noq_nn_results': pp_occmean_noq_nn_results,
                  'pp_occmean_onlyq_lm_results': pp_occmean_onlyq_lm_results,
                  'pp_occp95_basicq_lm_results': pp_occp95_basicq_lm_results,
                  'pp_occp95_noq_lm_results': pp_occp95_noq_lm_results,
                  'pp_occp95_basicq_lassocv_results': pp_occp95_basicq_lassocv_results,
                  'pp_occp95_noq_lassocv_results': pp_occp95_noq_lassocv_results,
                  'pp_occp95_basicq_poly_results': pp_occp95_basicq_poly_results,
                  'pp_occp95_noq_poly_results': pp_occp95_noq_poly_results,
                  'pp_occp95_basicq_rf_results': pp_occp95_basicq_rf_results,
                  'pp_occp95_noq_rf_results': pp_occp95_noq_rf_results,
                  'pp_occp95_basicq_svr_results': pp_occp95_basicq_svr_results,
                  'pp_occp95_noq_svr_results': pp_occp95_noq_svr_results,
                  'pp_occp95_basicq_nn_results': pp_occp95_basicq_nn_results,
                  'pp_occp95_noq_nn_results': pp_occp95_noq_nn_results,
                  'pp_occmean_basicq_load_results': pp_occmean_basicq_load_results,
                  'pp_occp95_basicq_sqrtload_results': pp_occp95_basicq_sqrtload_results,
                  'pp_occmean_basicq_hgbr_results': pp_occmean_basicq_hgbr_results,
                  'pp_occp95_basicq_hgbr_results': pp_occp95_basicq_hgbr_results,
                  'pp_occp95_onlyq_lm_results': pp_occp95_onlyq_lm_results
                  }

    create_cv_plots(experiment, unit, pp_results, figures_path)
    create_coeff_plots(experiment, unit, pp_results, figures_path)

    metrics_df = create_metrics_df(pp_results)
    metrics_df.to_csv(metrics_path_filename, index=False)

    predictions_df = create_predictions_df(pp_results)
    predictions_df.to_csv(Path(output_path, f"{experiment}_{unit}_predictions.csv"), index=False)

    sys.setrecursionlimit(10000)
    # Pickle the results
    with open(pickle_path_filename, 'wb') as persisted_file:
        pickle.dump(pp_results, persisted_file)

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
