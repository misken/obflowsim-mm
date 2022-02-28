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

    X_ldr_probblockedbypp_onlyq = \
        pd.read_csv(Path(data_path, f'X_ldr_probblockedbypp_onlyq_{exp}.csv'), index_col=0)
    X_ldr_probblockedbypp_onlyq_cols = X_ldr_probblockedbypp_onlyq.columns.tolist()

    X_ldr_condmeantimeblockedbypp_onlyq = \
        pd.read_csv(Path(data_path, f'X_ldr_condmeantimeblockedbypp_onlyq_{exp}.csv'), index_col=0)
    X_ldr_condmeantimeblockedbypp_onlyq_cols = X_ldr_condmeantimeblockedbypp_onlyq.columns.tolist()

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

    X_obs_probblockedbyldr_onlyq = pd.read_csv(Path(data_path, f'X_obs_probblockedbyldr_onlyq_{exp}.csv'),
                                                 index_col=0)
    X_obs_probblockedbyldr_onlyq_cols = X_obs_probblockedbyldr_onlyq.columns.tolist()

    X_obs_condmeantimeblockedbyldr_onlyq = \
        pd.read_csv(Path(data_path, f'X_obs_condmeantimeblockedbyldr_onlyq_{exp}.csv'), index_col=0)
    X_obs_condmeantimeblockedbyldr_onlyq_cols = X_obs_condmeantimeblockedbyldr_onlyq.columns.tolist()

    # Create dataframe consistent with X_ldr_q to be used for predictions
    X_ldr_q_scenarios_df = scenarios_df.loc[:, X_ldr_q_cols]
    X_ldr_basicq_scenarios_df = scenarios_df.loc[:, X_ldr_basicq_cols]
    X_ldr_noq_scenarios_df = scenarios_df.loc[:, X_ldr_noq_cols]
    X_ldr_occmean_onlyq_df = scenarios_df.loc[:, X_ldr_occmean_onlyq_cols]
    X_ldr_occp95_onlyq_df = scenarios_df.loc[:, X_ldr_occp95_onlyq_cols]
    X_ldr_probblockedbypp_onlyq_df = scenarios_df.loc[:, X_ldr_probblockedbypp_onlyq_cols]
    X_ldr_condmeantimeblockedbypp_onlyq_df = scenarios_df.loc[:, X_ldr_condmeantimeblockedbypp_onlyq_cols]
    X_pp_basicq_scenarios_df = scenarios_df.loc[:, X_pp_basicq_cols]
    X_pp_noq_scenarios_df = scenarios_df.loc[:, X_pp_noq_cols]
    X_pp_occmean_onlyq_df = scenarios_df.loc[:, X_pp_occmean_onlyq_cols]
    X_pp_occp95_onlyq_df = scenarios_df.loc[:, X_pp_occp95_onlyq_cols]

    X_obs_q_scenarios_df = scenarios_df.loc[:, X_obs_q_cols]
    X_obs_basicq_scenarios_df = scenarios_df.loc[:, X_obs_basicq_cols]
    X_obs_noq_scenarios_df = scenarios_df.loc[:, X_obs_noq_cols]
    X_obs_occmean_onlyq_df = scenarios_df.loc[:, X_obs_occmean_onlyq_cols]
    X_obs_occp95_onlyq_df = scenarios_df.loc[:, X_obs_occp95_onlyq_cols]
    X_obs_probblockedbyldr_onlyq_df = scenarios_df.loc[:, X_obs_probblockedbyldr_onlyq_cols]
    X_obs_condmeantimeblockedbyldr_onlyq_df = scenarios_df.loc[:, X_obs_condmeantimeblockedbyldr_onlyq_cols]

    return {'X_ldr_q': X_ldr_q_scenarios_df,
            'X_ldr_basicq': X_ldr_basicq_scenarios_df,
            'X_ldr_noq': X_ldr_noq_scenarios_df,
            'X_ldr_occmean_onlyq': X_ldr_occmean_onlyq_df,
            'X_ldr_occp95_onlyq': X_ldr_occp95_onlyq_df,
            'X_ldr_probblockedbypp_onlyq': X_ldr_probblockedbypp_onlyq_df,
            'X_ldr_condmeantimeblockedbypp_onlyq': X_ldr_condmeantimeblockedbypp_onlyq_df,
            'X_pp_basicq': X_pp_basicq_scenarios_df,
            'X_pp_noq': X_pp_noq_scenarios_df,
            'X_pp_occmean_onlyq': X_pp_occmean_onlyq_df,
            'X_pp_occp95_onlyq': X_pp_occp95_onlyq_df,
            'X_obs_q': X_obs_q_scenarios_df,
            'X_obs_basicq': X_obs_basicq_scenarios_df,
            'X_obs_noq': X_obs_noq_scenarios_df,
            'X_obs_occmean_onlyq': X_obs_occmean_onlyq_df,
            'X_obs_occp95_onlyq': X_obs_occp95_onlyq_df,
            'X_obs_probblockedbyldr_onlyq': X_obs_probblockedbyldr_onlyq_df,
            'X_obs_condmeantimeblockedbyldr_onlyq': X_obs_condmeantimeblockedbyldr_onlyq_df
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
        scenario_input_path_filename = Path(f'input/{perf_curve_scenarios_suffix}/exp15_obflowsim_scenario_inputs.csv')
        # Paths to mm model fitting pkl files (input)
        obs_pkl_path_filename = Path(f'output/{mm_experiment_suffix}/{mm_experiment_suffix}_obs_results.pkl')
        ldr_pkl_path_filename = Path(f'output/{mm_experiment_suffix}/{mm_experiment_suffix}_ldr_results.pkl')
        pp_pkl_path_filename = Path(f'output/{mm_experiment_suffix}/{mm_experiment_suffix}_pp_results.pkl')
        # Path for where X and y matix data will be written (output)
        matrix_data_path = Path(f'input/{mm_experiment_suffix}')

        # Read scenario input file
        input_scenarios_df = pd.read_csv(scenario_input_path_filename)

        # Create dictionary of X matrices to act as inputs for generating perf curves
        scenarios_dfs = make_x_scenarios(input_scenarios_df, mm_experiment_suffix, matrix_data_path)

        # Get results from pickle file
        with open(ldr_pkl_path_filename, 'rb') as pickle_file:
            ldr_results = pickle.load(pickle_file)

        with open(pp_pkl_path_filename, 'rb') as pickle_file:
            pp_results = pickle.load(pickle_file)

        with open(obs_pkl_path_filename, 'rb') as pickle_file:
            obs_results = pickle.load(pickle_file)

        # Make the predictions for each scenario - using OBS as base since has all inputs
        scenarios_io_df = scenarios_dfs['X_obs_q'].copy()

        # LDR
        features_models_ldr = [('q', 'lm'), ('basicq', 'lm'), ('noq', 'lm'),
                               ('q', 'lassocv'),
                               ('noq', 'poly'), ('basicq', 'poly'),
                               ('noq', 'rf'), ('basicq', 'rf'), ('q', 'rf'),
                               ('noq', 'nn'), ('basicq', 'nn'), ('q', 'nn')]

        for f, m in features_models_ldr:
            scenarios_io_df[f'pred_ldr_occmean_{f}_{m}'] = \
                ldr_results[f'ldr_occmean_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_ldr_{f}'])

            scenarios_io_df[f'pred_ldr_occp95_{f}_{m}'] = \
                ldr_results[f'ldr_occp95_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_ldr_{f}'])

            scenarios_io_df[f'pred_probblockedbypp_{f}_{m}'] = \
                ldr_results[f'probblockedbypp_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_ldr_{f}'])

            scenarios_io_df[f'pred_condmeantimeblockedbypp_{f}_{m}'] = \
                ldr_results[f'condmeantimeblockedbypp_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_ldr_{f}'])

        scenarios_io_df['pred_ldr_occmean_onlyq_lm'] = \
            ldr_results['ldr_occmean_onlyq_lm_results']['model'].predict(scenarios_dfs['X_ldr_occmean_onlyq'])

        scenarios_io_df['pred_ldr_occp95_onlyq_lm'] = \
            ldr_results['ldr_occp95_onlyq_lm_results']['model'].predict(scenarios_dfs['X_ldr_occp95_onlyq'])

        scenarios_io_df['pred_probblockedbypp_onlyq_lm'] = \
            ldr_results['probblockedbypp_onlyq_lm_results']['model'].predict(
                scenarios_dfs['X_ldr_probblockedbypp_onlyq'])

        scenarios_io_df['pred_probblockedbypp_q_erlangc'] = \
            ldr_results['probblockedbypp_q_erlangc_results']['model'].predict(
                scenarios_dfs['X_ldr_q'])

        scenarios_io_df['pred_condmeantimeblockedbypp_onlyq_lm'] = \
            ldr_results['condmeantimeblockedbypp_onlyq_lm_results']['model'].predict(
                scenarios_dfs['X_ldr_condmeantimeblockedbypp_onlyq'])

        scenarios_io_df['pred_condmeantimeblockedbypp_q_mgs'] = \
            ldr_results['condmeantimeblockedbypp_q_mgc_results']['model'].predict(
                scenarios_dfs['X_ldr_q'])

        print('done with new style predictions for LDR')

        # PP predictions
        features_models_pp = [('basicq', 'lm'), ('noq', 'lm'),
                              ('basicq', 'lassocv'),
                              ('noq', 'poly'), ('basicq', 'poly'),
                              ('noq', 'rf'), ('basicq', 'rf'),
                              ('noq', 'nn'), ('basicq', 'nn')]

        for f, m in features_models_pp:
            scenarios_io_df[f'pred_pp_occmean_{f}_{m}'] = \
                pp_results[f'pp_occmean_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_pp_{f}'])

            scenarios_io_df[f'pred_pp_occp95_{f}_{m}'] = \
                pp_results[f'pp_occp95_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_pp_{f}'])

        # onlyq pp
        scenarios_io_df['pred_pp_occmean_onlyq_lm'] = \
            pp_results['pp_occmean_onlyq_lm_results']['model'].predict(scenarios_dfs['X_pp_occmean_onlyq'])
        scenarios_io_df['pred_pp_occp95_onlyq_lm'] = \
            pp_results['pp_occp95_onlyq_lm_results']['model'].predict(scenarios_dfs['X_pp_occp95_onlyq'])

        print('done with new style predictions for PP')

        # OBS predictions
        features_models_obs = [('q', 'lm'), ('basicq', 'lm'), ('noq', 'lm'),
                               ('q', 'lassocv'),
                               ('noq', 'poly'), ('basicq', 'poly'),
                               ('noq', 'rf'), ('basicq', 'rf'), ('q', 'rf'),
                               ('noq', 'nn'), ('basicq', 'nn'), ('q', 'nn')]

        for f, m in features_models_obs:
            scenarios_io_df[f'pred_obs_occmean_{f}_{m}'] = \
                obs_results[f'obs_occmean_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_obs_{f}'])

            scenarios_io_df[f'pred_obs_occp95_{f}_{m}'] = \
                obs_results[f'obs_occp95_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_obs_{f}'])

            scenarios_io_df[f'pred_probblockedbyldr_{f}_{m}'] = \
                obs_results[f'probblockedbyldr_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_obs_{f}'])

            scenarios_io_df[f'pred_condmeantimeblockedbyldr_{f}_{m}'] = \
                obs_results[f'condmeantimeblockedbyldr_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_obs_{f}'])

        scenarios_io_df['pred_obs_occmean_onlyq_lm'] = \
            obs_results['obs_occmean_onlyq_lm_results']['model'].predict(scenarios_dfs['X_obs_occmean_onlyq'])

        scenarios_io_df['pred_obs_occp95_onlyq_lm'] = \
            obs_results['obs_occp95_onlyq_lm_results']['model'].predict(scenarios_dfs['X_obs_occp95_onlyq'])

        scenarios_io_df['pred_probblockedbyldr_onlyq_lm'] = \
            obs_results['probblockedbyldr_onlyq_lm_results']['model'].predict(
                scenarios_dfs['X_obs_probblockedbyldr_onlyq'])

        scenarios_io_df['pred_probblockedbyldr_q_erlangc'] = \
            obs_results['probblockedbyldr_q_erlangc_results']['model'].predict(
                scenarios_dfs['X_obs_q'])

        scenarios_io_df['pred_condmeantimeblockedbyldr_onlyq_lm'] = \
            obs_results['condmeantimeblockedbyldr_onlyq_lm_results']['model'].predict(
                scenarios_dfs['X_obs_condmeantimeblockedbyldr_onlyq'])

        print('done with new style predictions for OBS')

        scenarios_io_df.to_csv(path_scenario_csv, index=True)
        print(f'scenarios_io_df written to {path_scenario_csv}')


