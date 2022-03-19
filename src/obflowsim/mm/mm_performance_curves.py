from pathlib import Path
import pickle
import itertools

import numpy as np
import pandas as pd
import yaml

from mm_dataprep import qng_approx_from_inputs
from qng import qng


def make_x_scenarios(scenarios_df, exp, data_path):
    """
    Generate performance curve dataframes consistent with the X matrices used in mm fitting

    Parameters
    ----------
    scenarios_df : pandas Dataframe
        Base inputs for the performance curves. These should be consistent with the column names used in the
        X matrices used for mm fitting.
    exp : str
        Experiment id from original simulation runs from which metamodels were fitted
    data_path
        Location of X matrices files. Need to get the column specs from them so that we can create performance
        curve dataframes that can be used to generate predictions from previously fitted metamodels.

    Returns
    -------

    """

    units = ['obs', 'ldr', 'pp']

    # Read X matrices just to extract list of column names. These include both base and derived inputs.
    X_ldr_q = pd.read_csv(Path(data_path, f'X_ldr_q_{exp}.csv'), index_col=0)
    X_ldr_q_cols = X_ldr_q.columns.tolist()

    X_ldr_basicq = pd.read_csv(Path(data_path, f'X_ldr_basicq_{exp}.csv'), index_col=0)
    X_ldr_basicq_cols = X_ldr_basicq.columns.tolist()

    X_ldr_noq = pd.read_csv(Path(data_path, f'X_ldr_noq_{exp}.csv'), index_col=0)
    X_ldr_noq_cols = X_ldr_noq.columns.tolist()

    X_ldr_occmean_onlyq = pd.read_csv(Path(data_path, f'X_ldr_occmean_onlyq_{exp}.csv'), index_col=0)
    X_ldr_occmean_onlyq_cols = X_ldr_occmean_onlyq.columns.tolist()

    X_ldr_occp95_onlyq = pd.read_csv(Path(data_path, f'X_ldr_occp95_onlyq_{exp}.csv'), index_col=0)
    X_ldr_occp95_onlyq_cols = X_ldr_occp95_onlyq.columns.tolist()

    X_ldr_probblocked_onlyq = \
        pd.read_csv(Path(data_path, f'X_ldr_probblocked_onlyq_{exp}.csv'), index_col=0)
    X_ldr_probblocked_onlyq_cols = X_ldr_probblocked_onlyq.columns.tolist()

    X_ldr_condmeantimeblocked_onlyq = \
        pd.read_csv(Path(data_path, f'X_ldr_condmeantimeblocked_onlyq_{exp}.csv'), index_col=0)
    X_ldr_condmeantimeblocked_onlyq_cols = X_ldr_condmeantimeblocked_onlyq.columns.tolist()

    X_pp_basicq = pd.read_csv(Path(data_path, f'X_pp_basicq_{exp}.csv'), index_col=0)
    X_pp_basicq_cols = X_pp_basicq.columns.tolist()

    X_pp_noq = pd.read_csv(Path(data_path, f'X_pp_noq_{exp}.csv'), index_col=0)
    X_pp_noq_cols = X_pp_noq.columns.tolist()

    X_pp_occp95_onlyq = pd.read_csv(Path(data_path, f'X_pp_occp95_onlyq_{exp}.csv'), index_col=0)
    X_pp_occp95_onlyq_cols = X_pp_occp95_onlyq.columns.tolist()

    X_pp_occmean_onlyq = pd.read_csv(Path(data_path, f'X_pp_occmean_onlyq_{exp}.csv'), index_col=0)
    X_pp_occmean_onlyq_cols = X_pp_occmean_onlyq.columns.tolist()

    # OBS
    X_obs_q = pd.read_csv(Path(data_path, f'X_obs_q_{exp}.csv'), index_col=0)
    X_obs_q_cols = X_obs_q.columns.tolist()

    X_obs_basicq = pd.read_csv(Path(data_path, f'X_obs_basicq_{exp}.csv'), index_col=0)
    X_obs_basicq_cols = X_obs_basicq.columns.tolist()

    X_obs_noq = pd.read_csv(Path(data_path, f'X_obs_noq_{exp}.csv'), index_col=0)
    X_obs_noq_cols = X_obs_noq.columns.tolist()

    X_obs_occmean_onlyq = pd.read_csv(Path(data_path, f'X_obs_occmean_onlyq_{exp}.csv'), index_col=0)
    X_obs_occmean_onlyq_cols = X_obs_occmean_onlyq.columns.tolist()

    X_obs_occp95_onlyq = pd.read_csv(Path(data_path, f'X_obs_occp95_onlyq_{exp}.csv'), index_col=0)
    X_obs_occp95_onlyq_cols = X_obs_occp95_onlyq.columns.tolist()

    X_obs_probblocked_onlyq = pd.read_csv(Path(data_path, f'X_obs_probblocked_onlyq_{exp}.csv'),
                                                 index_col=0)
    X_obs_probblocked_onlyq_cols = X_obs_probblocked_onlyq.columns.tolist()

    X_obs_condmeantimeblocked_onlyq = \
        pd.read_csv(Path(data_path, f'X_obs_condmeantimeblocked_onlyq_{exp}.csv'), index_col=0)
    X_obs_condmeantimeblocked_onlyq_cols = X_obs_condmeantimeblocked_onlyq.columns.tolist()

    # Create dataframe consistent with X_ldr_q to be used for predictions
    X_ldr_q_scenarios_df = scenarios_df.loc[:, X_ldr_q_cols]
    X_ldr_basicq_scenarios_df = scenarios_df.loc[:, X_ldr_basicq_cols]
    X_ldr_noq_scenarios_df = scenarios_df.loc[:, X_ldr_noq_cols]
    X_ldr_occmean_onlyq_df = scenarios_df.loc[:, X_ldr_occmean_onlyq_cols]
    X_ldr_occp95_onlyq_df = scenarios_df.loc[:, X_ldr_occp95_onlyq_cols]
    X_ldr_probblocked_onlyq_df = scenarios_df.loc[:, X_ldr_probblocked_onlyq_cols]
    X_ldr_condmeantimeblocked_onlyq_df = scenarios_df.loc[:, X_ldr_condmeantimeblocked_onlyq_cols]
    X_pp_basicq_scenarios_df = scenarios_df.loc[:, X_pp_basicq_cols]
    X_pp_noq_scenarios_df = scenarios_df.loc[:, X_pp_noq_cols]
    X_pp_occmean_onlyq_df = scenarios_df.loc[:, X_pp_occmean_onlyq_cols]
    X_pp_occp95_onlyq_df = scenarios_df.loc[:, X_pp_occp95_onlyq_cols]

    X_obs_q_scenarios_df = scenarios_df.loc[:, X_obs_q_cols]
    X_obs_basicq_scenarios_df = scenarios_df.loc[:, X_obs_basicq_cols]
    X_obs_noq_scenarios_df = scenarios_df.loc[:, X_obs_noq_cols]
    X_obs_occmean_onlyq_df = scenarios_df.loc[:, X_obs_occmean_onlyq_cols]
    X_obs_occp95_onlyq_df = scenarios_df.loc[:, X_obs_occp95_onlyq_cols]
    X_obs_probblocked_onlyq_df = scenarios_df.loc[:, X_obs_probblocked_onlyq_cols]
    X_obs_condmeantimeblocked_onlyq_df = scenarios_df.loc[:, X_obs_condmeantimeblocked_onlyq_cols]

    return {'X_ldr_q': X_ldr_q_scenarios_df,
            'X_ldr_basicq': X_ldr_basicq_scenarios_df,
            'X_ldr_noq': X_ldr_noq_scenarios_df,
            'X_ldr_occmean_onlyq': X_ldr_occmean_onlyq_df,
            'X_ldr_occp95_onlyq': X_ldr_occp95_onlyq_df,
            'X_ldr_probblocked_onlyq': X_ldr_probblocked_onlyq_df,
            'X_ldr_condmeantimeblocked_onlyq': X_ldr_condmeantimeblocked_onlyq_df,
            'X_pp_basicq': X_pp_basicq_scenarios_df,
            'X_pp_noq': X_pp_noq_scenarios_df,
            'X_pp_occmean_onlyq': X_pp_occmean_onlyq_df,
            'X_pp_occp95_onlyq': X_pp_occp95_onlyq_df,
            'X_obs_q': X_obs_q_scenarios_df,
            'X_obs_basicq': X_obs_basicq_scenarios_df,
            'X_obs_noq': X_obs_noq_scenarios_df,
            'X_obs_occmean_onlyq': X_obs_occmean_onlyq_df,
            'X_obs_occp95_onlyq': X_obs_occp95_onlyq_df,
            'X_obs_probblocked_onlyq': X_obs_probblocked_onlyq_df,
            'X_obs_condmeantimeblocked_onlyq': X_obs_condmeantimeblocked_onlyq_df
            }


def hyper_erlang_cv2(means, stages, probs):
    mean = np.dot(means, probs)
    rates = [1 / mu for mu in means]
    m2 = qng.hyper_erlang_moment(rates, stages, probs, 2)
    var = m2 - mean ** 2
    cv2 = var / mean ** 2
    return cv2

def process_command_line():
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='mm_merge_predict_simulated',
                                     description='merge simulation scenario output with mm predictions')

    # Add arguments
    parser.add_argument(
        "mm_experiment", type=str,
        help="Experiment used to fit metamodels"
    )

    parser.add_argument(
        "predict_experiment", type=str,
        help="Experiment for which to predict"
    )

    parser.add_argument(
        "scenario_input_path_filename", type=str,
        help="Path to csv file which contains scenario inputs"
    )

    parser.add_argument(
        "y_data_path", type=str,
        help="Path to directory containing y data files (which are created from sim output)"
    )

    parser.add_argument(
        "output_filename", type=str,
        help="Path to merged output csv file"
    )

    # Do the parsing and return the populated namespace with the input arg values
    args = parser.parse_args()
    return args


def main(argv=None):
    # Parse command line arguments
    mm_args = process_command_line()
    mm_exp = mm_args.mm_experiment_suffix
    pred_exp = mm_args.predict_experiment
    scenario_input_path_filename = mm_args.scenario_input_path_filename
    
    y_data_path = mm_args.y_data_path
    output_filename = mm_args.output_filename


if __name__ == '__main__':

    override_args = True

    if override_args:
        mm_experiment_suffix = 'exp13'
        perf_curve_scenarios_suffix = 'exp16'
        # Path to scenario input csv file created by scenario_tools.py (input)
        scenario_input_path_filename = \
            Path(f'input/{perf_curve_scenarios_suffix}/{perf_curve_scenarios_suffix}_obflowsim_scenario_inputs.csv')
        # Paths to mm model fitting pkl files (input)
        obs_pkl_path_filename = Path(f'output/{mm_experiment_suffix}/{mm_experiment_suffix}_obs_results.pkl')
        ldr_pkl_path_filename = Path(f'output/{mm_experiment_suffix}/{mm_experiment_suffix}_ldr_results.pkl')
        pp_pkl_path_filename = Path(f'output/{mm_experiment_suffix}/{mm_experiment_suffix}_pp_results.pkl')
        # Path where X and y matrix data from fitted models (input)
        matrix_data_path = Path(f'input/{mm_experiment_suffix}')
        # Path to output csv file
        performance_curve_predictions_path_filename = \
            Path(f'output/{perf_curve_scenarios_suffix}/pc_predictions_{perf_curve_scenarios_suffix}.csv')

        performance_curve_predictions_long_path_filename = \
            Path(f'output/{perf_curve_scenarios_suffix}/pc_predictions_{perf_curve_scenarios_suffix}_long.csv')

        # Read scenario input file
        input_scenarios_df = pd.read_csv(scenario_input_path_filename)

        # Create dictionary of X matrices to act as inputs for generating perf curves
        scenarios_x_dfs = make_x_scenarios(input_scenarios_df, mm_experiment_suffix, matrix_data_path)

        # Get results from pickle file
        with open(ldr_pkl_path_filename, 'rb') as pickle_file:
            ldr_results = pickle.load(pickle_file)

        with open(pp_pkl_path_filename, 'rb') as pickle_file:
            pp_results = pickle.load(pickle_file)

        with open(obs_pkl_path_filename, 'rb') as pickle_file:
            obs_results = pickle.load(pickle_file)

        fit_results = {'ldr': ldr_results,
                       'pp': pp_results,
                       'obs': obs_results}

        # Make the predictions for each scenario - using OBS as base since has all inputs
        scenarios_predictions_df = scenarios_x_dfs['X_obs_q'].copy()

        # Add scenario column
        num_scenarios = len(scenarios_predictions_df)
        scenarios_predictions_df['scenario'] = np.arange(1, num_scenarios + 1)
        cols = scenarios_predictions_df.columns.to_list()
        new_col_order = cols[-1:] + cols[:-1]
        scenarios_predictions_df = scenarios_predictions_df[new_col_order]

        features_models = [('q', 'lm'), ('basicq', 'lm'), ('noq', 'lm'),
                               ('q', 'lassocv'), ('basicq', 'lassocv'), ('noq', 'lassocv'),
                               ('q', 'poly'), ('basicq', 'poly'), ('noq', 'poly'),
                               ('q', 'rf'), ('basicq', 'rf'), ('noq', 'rf'),
                               ('q', 'nn'), ('basicq', 'nn'), ('noq', 'nn'),
                               ('onlyq', 'lm'),
                               ('q', 'load'), ('q', 'sqrtload'),
                               ('q', 'erlangc'), ('q', 'mgc')]

        unit_predictions = []

        for unit in ['obs', 'ldr', 'pp']:
            unit_fit_results = fit_results[unit]
            for f, m in features_models:
                for measure in ['occmean', 'occp95', 'probblocked', 'condmeantimeblocked']:

                    # Check if this is a modeled combination
                    if f'{unit}_{measure}_{f}_{m}_results' in unit_fit_results:

                        # Handle case of onlyq
                        if f == 'onlyq':
                            fullfeature = f'{measure}_{f}'
                        else:
                            fullfeature = f

                        scenarios_predictions_df[f'pred_{unit}_{measure}_{f}_{m}'] = \
                            unit_fit_results[f'{unit}_{measure}_{f}_{m}_results']['model'].predict(scenarios_x_dfs[f'X_{unit}_{fullfeature}'])

                        result = {'unit': unit, 'qdata': f, 'model': m,
                                  'measure': measure,
                                  'scenarios': scenarios_predictions_df['scenario'],
                                  'predictions': scenarios_predictions_df[f'pred_{unit}_{measure}_{f}_{m}']}

                        unit_predictions.append(result)

            print(f'done with predictions for {unit}')

        unit_predictions_tuples = [(s, result['unit'], result['measure'], result['qdata'], result['model'], p)
                                  for result in unit_predictions
                                  for (s, p) in zip(result['scenarios'], result['predictions'])]

        unit_predictions_long_df = pd.DataFrame(unit_predictions_tuples,
                                               columns=['scenario', 'unit', 'measure', 'qdata', 'model', 'prediction'])

        unit_predictions_long_df.to_csv(performance_curve_predictions_long_path_filename, index=False)

        # Export wide dataframe
        scenarios_predictions_df.to_csv(performance_curve_predictions_path_filename, index=False)
        print(f'scenarios_predictions_df written to {performance_curve_predictions_path_filename}')
        print(f'scenarios_predictions_long_df written to {performance_curve_predictions_long_path_filename}')


