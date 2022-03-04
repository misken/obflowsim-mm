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


if __name__ == '__main__':

    override_args = True

    if override_args:
        mm_experiment_suffix = 'exp13'
        perf_curve_scenarios_suffix = 'exp15'
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



        # Queueing approximation based models for ldr
        # scenarios_predictions_df['pred_ldr_occmean_onlyq_lm'] = \
        #     ldr_results['ldr_occmean_onlyq_lm_results']['model'].predict(scenarios_x_dfs['X_ldr_occmean_onlyq'])
        #
        # scenarios_predictions_df['pred_ldr_occp95_onlyq_lm'] = \
        #     ldr_results['ldr_occp95_onlyq_lm_results']['model'].predict(scenarios_x_dfs['X_ldr_occp95_onlyq'])
        #
        # scenarios_predictions_df['pred_probblocked_onlyq_lm'] = \
        #     ldr_results['ldr_probblocked_onlyq_lm_results']['model'].predict(
        #         scenarios_x_dfs['X_ldr_probblocked_onlyq'])
        #
        # scenarios_predictions_df['pred_probblocked_q_erlangc'] = \
        #     ldr_results['ldr_probblocked_q_erlangc_results']['model'].predict(
        #         scenarios_x_dfs['X_ldr_q'])
        #
        # scenarios_predictions_df['pred_condmeantimeblocked_onlyq_lm'] = \
        #     ldr_results['ldr_condmeantimeblocked_onlyq_lm_results']['model'].predict(
        #         scenarios_x_dfs['X_ldr_condmeantimeblocked_onlyq'])
        #
        # scenarios_predictions_df['pred_condmeantimeblocked_q_mgs'] = \
        #     ldr_results['ldr_condmeantimeblocked_q_mgc_results']['model'].predict(
        #         scenarios_x_dfs['X_ldr_q'])

        # Create long dataframe from results list
        # First create list of tuples
        unit_predictions_tuples = [(s, result['unit'], result['measure'], result['qdata'], result['model'], p)
                                  for result in unit_predictions
                                  for (s, p) in zip(result['scenarios'], result['predictions'])]

        unit_predictions_long_df = pd.DataFrame(unit_predictions_tuples,
                                               columns=['scenario', 'unit', 'measure', 'qdata', 'model', 'prediction'])



        # PP predictions
        # features_models_pp = [('basicq', 'lm'), ('noq', 'lm'),
        #                        ('basicq', 'lassocv'), ('noq', 'lassocv'),
        #                        ('basicq', 'poly'), ('noq', 'poly'),
        #                        ('basicq', 'rf'), ('noq', 'rf'),
        #                        ('basicq', 'nn'), ('noq', 'nn')]
        #
        # pp_predictions = []
        #
        # for f, m in features_models_pp:
        #     for measure in ['occmean', 'occp95']:
        #         scenarios_predictions_df[f'pred_pp_{measure}_{f}_{m}'] = \
        #             pp_results[f'pp_{measure}_{f}_{m}_results']['model'].predict(scenarios_x_dfs[f'X_pp_{f}'])
        #
        #         result = {'qdata': f, 'model': m,
        #                   'measure': measure,
        #                   'scenarios': scenarios_predictions_df['scenario'],
        #                   'predictions': scenarios_predictions_df[f'pred_pp_{measure}_{f}_{m}']}
        #
        #         pp_predictions.append(result)
        #
        # pp_predictions_tuples = [(s, 'pp', result['measure'], result['qdata'], result['model'], p)
        #                           for result in pp_predictions
        #                           for (s, p) in zip(result['scenarios'], result['predictions'])]
        #
        # pp_predictions_long_df = pd.DataFrame(pp_predictions_tuples,
        #                                        columns=['scenario', 'unit', 'measure', 'qdata', 'model', 'prediction'])
        #
        # # onlyq pp
        # scenarios_predictions_df['pred_pp_occmean_onlyq_lm'] = \
        #     pp_results['pp_occmean_onlyq_lm_results']['model'].predict(scenarios_x_dfs['X_pp_occmean_onlyq'])
        # scenarios_predictions_df['pred_pp_occp95_onlyq_lm'] = \
        #     pp_results['pp_occp95_onlyq_lm_results']['model'].predict(scenarios_x_dfs['X_pp_occp95_onlyq'])
        #
        # print('done with new style predictions for PP')

        # OBS predictions
        # features_models_obs = [('q', 'lm'), ('basicq', 'lm'), ('noq', 'lm'),
        #                        ('q', 'lassocv'), ('basicq', 'lassocv'), ('noq', 'lassocv'),
        #                        ('q', 'poly'), ('basicq', 'poly'), ('noq', 'poly'),
        #                        ('q', 'rf'), ('basicq', 'rf'), ('noq', 'rf'),
        #                        ('q', 'nn'), ('basicq', 'nn'), ('noq', 'nn')]
        #
        # obs_predictions = []
        #
        # for f, m in features_models_obs:
        #     for measure in ['occmean', 'occp95', 'probblocked', 'condmeantimeblocked']:
        #
        #         scenarios_predictions_df[f'pred_obs_{measure}_{f}_{m}'] = \
        #             obs_results[f'obs_{measure}_{f}_{m}_results']['model'].predict(scenarios_x_dfs[f'X_obs_{f}'])
        #
        #         result = {'qdata': f, 'model': m,
        #                   'measure': measure,
        #                   'scenarios': scenarios_predictions_df['scenario'],
        #                   'predictions': scenarios_predictions_df[f'pred_obs_{measure}_{f}_{m}']}
        #
        #         obs_predictions.append(result)
        #
        # obs_predictions_tuples = [(s, 'obs', result['measure'], result['qdata'], result['model'], p)
        #                           for result in obs_predictions
        #                           for (s, p) in zip(result['scenarios'], result['predictions'])]
        #
        # obs_predictions_long_df = pd.DataFrame(obs_predictions_tuples,
        #                                        columns=['scenario', 'unit', 'measure', 'qdata', 'model', 'prediction'])
        #
        # scenarios_predictions_df['pred_obs_occmean_onlyq_lm'] = \
        #     obs_results['obs_occmean_onlyq_lm_results']['model'].predict(scenarios_x_dfs['X_obs_occmean_onlyq'])
        #
        # scenarios_predictions_df['pred_obs_occp95_onlyq_lm'] = \
        #     obs_results['obs_occp95_onlyq_lm_results']['model'].predict(scenarios_x_dfs['X_obs_occp95_onlyq'])
        #
        # scenarios_predictions_df['pred_probblocked_onlyq_lm'] = \
        #     obs_results['obs_probblocked_onlyq_lm_results']['model'].predict(
        #         scenarios_x_dfs['X_obs_probblocked_onlyq'])
        #
        # scenarios_predictions_df['pred_probblocked_q_erlangc'] = \
        #     obs_results['obs_probblocked_q_erlangc_results']['model'].predict(
        #         scenarios_x_dfs['X_obs_q'])
        #
        # scenarios_predictions_df['pred_condmeantimeblocked_onlyq_lm'] = \
        #     obs_results['obs_condmeantimeblocked_onlyq_lm_results']['model'].predict(
        #         scenarios_x_dfs['X_obs_condmeantimeblocked_onlyq'])
        #
        # print('done with new style predictions for OBS')

        # Concat the unit specific long dataframes
        # scenarios_predictions_long_df = pd.concat([ldr_predictions_long_df,
        #                                            pp_predictions_long_df, obs_predictions_long_df])

        unit_predictions_long_df.to_csv(performance_curve_predictions_long_path_filename, index=False)

        # Export wide dataframe
        scenarios_predictions_df.to_csv(performance_curve_predictions_path_filename, index=False)
        print(f'scenarios_predictions_df written to {performance_curve_predictions_path_filename}')
        print(f'scenarios_predictions_long_df written to {performance_curve_predictions_long_path_filename}')


