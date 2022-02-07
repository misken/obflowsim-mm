from pathlib import Path
import pickle
import itertools

import numpy as np
import pandas as pd
import yaml

from obnetwork import obnetwork
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

    # List of column names from simulation meta_input file. These are just the base inputs - the derived
    # inputs are created below.
    meta_input_cols = scenarios_df.columns.tolist()

    units = ['obs', 'ldr', 'pp']

    # Read X matrices just to extract list of column names. These include both base and derived inputs.
    X_ldr_q = pd.read_csv(Path(data_path, f'X_ldr_q_{exp}.csv'), index_col=0)
    X_ldr_q_cols = X_ldr_q.columns.tolist()

    X_ldr_basicq = pd.read_csv(Path(data_path, f'X_ldr_basicq_{exp}.csv'), index_col=0)
    X_ldr_basicq_cols = X_ldr_basicq.columns.tolist()

    X_ldr_noq = pd.read_csv(Path(data_path, f'X_ldr_noq_{exp}.csv'), index_col=0)
    X_ldr_noq_cols = X_ldr_noq.columns.tolist()

    X_ldr_occmean_onlyq = pd.read_csv(Path(data_path, f'X_ldr_occmean_onlyq_exp11.csv'), index_col=0)
    X_ldr_occmean_onlyq_cols = X_ldr_occmean_onlyq.columns.tolist()

    X_ldr_occp95_onlyq = pd.read_csv(Path(data_path, f'X_ldr_occp95_onlyq_exp11.csv'), index_col=0)
    X_ldr_occp95_onlyq_cols = X_ldr_occp95_onlyq.columns.tolist()

    X_ldr_prob_blockedby_pp_onlyq = \
        pd.read_csv(Path(data_path, f'X_ldr_prob_blockedby_pp_onlyq_exp11.csv'), index_col=0)
    X_ldr_prob_blockedby_pp_onlyq_cols = X_ldr_prob_blockedby_pp_onlyq.columns.tolist()

    X_ldr_condmeantime_blockedby_pp_onlyq = \
        pd.read_csv(Path(data_path, f'X_ldr_condmeantime_blockedby_pp_onlyq_exp11.csv'), index_col=0)
    X_ldr_condmeantime_blockedby_pp_onlyq_cols = X_ldr_condmeantime_blockedby_pp_onlyq.columns.tolist()

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

    X_obs_prob_blockedby_ldr_onlyq = pd.read_csv(Path(data_path, f'X_obs_prob_blockedby_ldr_onlyq_{exp}.csv'),
                                                 index_col=0)
    X_obs_prob_blockedby_ldr_onlyq_cols = X_obs_prob_blockedby_ldr_onlyq.columns.tolist()

    X_obs_condmeantime_blockedby_ldr_onlyq = \
        pd.read_csv(Path(data_path, f'X_obs_condmeantime_blockedby_ldr_onlyq_{exp}.csv'), index_col=0)
    X_obs_condmeantime_blockedby_ldr_onlyq_cols = X_obs_condmeantime_blockedby_ldr_onlyq.columns.tolist()

    # Compute overall mean los and cv2 for PP and LDR
    scenarios_df['mean_los_pp'] = scenarios_df['c_sect_prob'] * scenarios_df['mean_los_pp_c'] + (
            1 - scenarios_df['c_sect_prob']) * scenarios_df['mean_los_pp_noc']

    # PP los is distributed as a hypererlang
    scenarios_df['pp_cv2_svctime'] = scenarios_df.apply(lambda x: hyper_erlang_cv2(
        [x.mean_los_pp_noc, x.mean_los_pp_c], [x.num_erlang_stages_pp, x.num_erlang_stages_pp],
        [1 - x.c_sect_prob, x.c_sect_prob]), axis=1)

    # LDR los is distributed as erlanglm
    scenarios_df['ldr_cv2_svctime'] = scenarios_df.apply(lambda x: 1 / x.num_erlang_stages_ldr, axis=1)
    scenarios_df['obs_cv2_svctime'] = scenarios_df.apply(lambda x: 1 / x.num_erlang_stages_obs, axis=1)

    # Compute load and rho related terms
    for unit in units:
        scenarios_df[f'load_{unit}'] = scenarios_df[f'arrival_rate'] * scenarios_df[f'mean_los_{unit}']
        scenarios_df[f'rho_{unit}'] = scenarios_df[f'load_{unit}'] / scenarios_df[f'cap_{unit}']
        scenarios_df[f'sqrt_load_{unit}'] = np.sqrt(scenarios_df[f'load_{unit}'])

    # Compute queueing approximation terms using obnetwork library
    scenarios_df['prob_blockedby_pp_approx'] = scenarios_df.apply(lambda x: obnetwork.prob_blockedby_pp_hat(
        x.arrival_rate, x.mean_los_pp, x.cap_pp, x.pp_cv2_svctime), axis=1)

    scenarios_df['condmeantime_blockedby_pp_approx'] = scenarios_df.apply(
        lambda x: obnetwork.condmeantime_blockedby_pp_hat(
            x.arrival_rate, x.mean_los_pp, x.cap_pp, x.pp_cv2_svctime), axis=1)

    # The next three derived inputs are all created by obnetwork.obs_blockedby_ldr_hats()
    scenarios_df['prob_blockedby_ldr_approx'] = \
        scenarios_df.apply(lambda x: obnetwork.obs_blockedby_ldr_hats(
            x.arrival_rate, x.c_sect_prob, x.mean_los_ldr, x.ldr_cv2_svctime, x.cap_ldr,
            x.mean_los_pp, x.pp_cv2_svctime, x.cap_pp)[2], axis=1)

    scenarios_df['condmeantime_blockedby_ldr_approx'] = \
        scenarios_df.apply(lambda x: obnetwork.obs_blockedby_ldr_hats(
            x.arrival_rate, x.c_sect_prob, x.mean_los_ldr, x.ldr_cv2_svctime, x.cap_ldr,
            x.mean_los_pp, x.pp_cv2_svctime, x.cap_pp)[3], axis=1)

    scenarios_df['ldr_effmean_svctime_approx'] = \
        scenarios_df.apply(lambda x: obnetwork.obs_blockedby_ldr_hats(
            x.arrival_rate, x.c_sect_prob, x.mean_los_ldr, x.ldr_cv2_svctime, x.cap_ldr,
            x.mean_los_pp, x.pp_cv2_svctime, x.cap_pp)[1], axis=1)

    scenarios_df['obs_effmean_svctime_approx'] = \
        scenarios_df['mean_los_pp'] + \
        scenarios_df['prob_blockedby_ldr_approx'] * scenarios_df['condmeantime_blockedby_ldr_approx']

    # Create effective load related terms using the approximation for ldr effective mean service time
    for unit in ['obs', 'ldr']:
        scenarios_df[f'{unit}_eff_load'] = scenarios_df['arrival_rate'] * scenarios_df[f'{unit}_effmean_svctime_approx']
        scenarios_df[f'{unit}_eff_sqrtload'] = np.sqrt(scenarios_df[f'{unit}_eff_load'])

    # Create dataframe consistent with X_ldr_q to be used for predictions
    X_ldr_q_scenarios_df = scenarios_df.loc[:, X_ldr_q_cols]
    X_ldr_basicq_scenarios_df = scenarios_df.loc[:, X_ldr_basicq_cols]
    X_ldr_noq_scenarios_df = scenarios_df.loc[:, X_ldr_noq_cols]
    X_ldr_occmean_onlyq_df = scenarios_df.loc[:, X_ldr_occmean_onlyq_cols]
    X_ldr_occp95_onlyq_df = scenarios_df.loc[:, X_ldr_occp95_onlyq_cols]
    X_ldr_prob_blockedby_pp_onlyq_df = scenarios_df.loc[:, X_ldr_prob_blockedby_pp_onlyq_cols]
    X_ldr_condmeantime_blockedby_pp_onlyq_df = scenarios_df.loc[:, X_ldr_condmeantime_blockedby_pp_onlyq_cols]
    X_pp_basicq_scenarios_df = scenarios_df.loc[:, X_pp_basicq_cols]
    X_pp_noq_scenarios_df = scenarios_df.loc[:, X_pp_noq_cols]
    X_pp_occmean_onlyq_df = scenarios_df.loc[:, X_pp_occmean_onlyq_cols]
    X_pp_occp95_onlyq_df = scenarios_df.loc[:, X_pp_occp95_onlyq_cols]

    X_obs_q_scenarios_df = scenarios_df.loc[:, X_obs_q_cols]
    X_obs_basicq_scenarios_df = scenarios_df.loc[:, X_obs_basicq_cols]
    X_obs_noq_scenarios_df = scenarios_df.loc[:, X_obs_noq_cols]
    X_obs_occmean_onlyq_df = scenarios_df.loc[:, X_obs_occmean_onlyq_cols]
    X_obs_occp95_onlyq_df = scenarios_df.loc[:, X_obs_occp95_onlyq_cols]
    X_obs_prob_blockedby_ldr_onlyq_df = scenarios_df.loc[:, X_obs_prob_blockedby_ldr_onlyq_cols]
    X_obs_condmeantime_blockedby_ldr_onlyq_df = scenarios_df.loc[:, X_obs_condmeantime_blockedby_ldr_onlyq_cols]

    # Create dataframe of the just the base input columns
    _meta_inputs_df = scenarios_df.loc[:, meta_input_cols]

    return {'X_ldr_q': X_ldr_q_scenarios_df,
            'X_ldr_basicq': X_ldr_basicq_scenarios_df,
            'X_ldr_noq': X_ldr_noq_scenarios_df,
            'X_ldr_occmean_onlyq': X_ldr_occmean_onlyq_df,
            'X_ldr_occp95_onlyq': X_ldr_occp95_onlyq_df,
            'X_ldr_prob_blockedby_pp_onlyq': X_ldr_prob_blockedby_pp_onlyq_df,
            'X_ldr_condmeantime_blockedby_pp_onlyq': X_ldr_condmeantime_blockedby_pp_onlyq_df,
            'X_pp_basicq': X_pp_basicq_scenarios_df,
            'X_pp_noq': X_pp_noq_scenarios_df,
            'X_pp_occmean_onlyq': X_pp_occmean_onlyq_df,
            'X_pp_occp95_onlyq': X_pp_occp95_onlyq_df,
            'meta_inputs': _meta_inputs_df,
            'X_obs_q': X_obs_q_scenarios_df,
            'X_obs_basicq': X_obs_basicq_scenarios_df,
            'X_obs_noq': X_obs_noq_scenarios_df,
            'X_obs_occmean_onlyq': X_obs_occmean_onlyq_df,
            'X_obs_occp95_onlyq': X_obs_occp95_onlyq_df,
            'X_obs_prob_blockedby_ldr_onlyq': X_obs_prob_blockedby_ldr_onlyq_df,
            'X_obs_condmeantime_blockedby_ldr_onlyq': X_obs_condmeantime_blockedby_ldr_onlyq_df
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
        mm_experiment_suffix = "exp11"
        perf_curve_scenarios_suffix = "exp11e"
        # Path to scenario yaml file created by scenario_grid.py
        path_scenario_grid_yaml = Path("mm_use", f"scenario_grid_{perf_curve_scenarios_suffix}.yaml")
        path_scenario_csv = Path("mm_use", f"X_performance_curves_{perf_curve_scenarios_suffix}.csv")
        siminout_path = Path("data/siminout")
        matrix_data_path = Path("data")

        with open(path_scenario_grid_yaml, "r") as stream:
            try:
                scenario_grid = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        print(scenario_grid)

        input_scenarios = [scn for scn in itertools.product(*[value for key, value in scenario_grid.items()])]

        cols = list(scenario_grid.keys())

        input_scenarios_df = pd.DataFrame(input_scenarios, columns=cols)
        num_scenarios = len(input_scenarios_df.index)
        input_scenarios_df.set_index(np.arange(1, num_scenarios + 1), inplace=True)
        input_scenarios_df.index.name = 'scenario'

        # Create dictionary of X matrices to act as inputs for generating perf curves
        scenarios_dfs = make_x_scenarios(input_scenarios_df, mm_experiment_suffix, matrix_data_path)

        # Get results from pickle file
        with open(Path("output", f"ldr_results_{mm_experiment_suffix}.pkl"), 'rb') as pickle_file:
            ldr_results = pickle.load(pickle_file)

        with open(Path("output", f"pp_results_{mm_experiment_suffix}.pkl"), 'rb') as pickle_file:
            pp_results = pickle.load(pickle_file)

        with open(Path("output", f"obs_results_{mm_experiment_suffix}.pkl"), 'rb') as pickle_file:
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
            scenarios_io_df[f'pred_ldr_occ_mean_{f}_{m}'] = \
                ldr_results[f'ldr_occ_mean_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_ldr_{f}'])

            scenarios_io_df[f'pred_ldr_occ_p95_{f}_{m}'] = \
                ldr_results[f'ldr_occ_p95_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_ldr_{f}'])

            scenarios_io_df[f'pred_prob_blockedby_pp_{f}_{m}'] = \
                ldr_results[f'prob_blockedby_pp_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_ldr_{f}'])

            scenarios_io_df[f'pred_condmeantime_blockedby_pp_{f}_{m}'] = \
                ldr_results[f'condmeantime_blockedby_pp_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_ldr_{f}'])

        scenarios_io_df['pred_ldr_occ_mean_onlyq_lm'] = \
            ldr_results['ldr_occ_mean_onlyq_lm_results']['model'].predict(scenarios_dfs['X_ldr_occmean_onlyq'])

        scenarios_io_df['pred_ldr_occ_p95_onlyq_lm'] = \
            ldr_results['ldr_occ_p95_onlyq_lm_results']['model'].predict(scenarios_dfs['X_ldr_occp95_onlyq'])

        scenarios_io_df['pred_prob_blockedby_pp_onlyq_lm'] = \
            ldr_results['prob_blockedby_pp_onlyq_lm_results']['model'].predict(
                scenarios_dfs['X_ldr_prob_blockedby_pp_onlyq'])

        scenarios_io_df['pred_prob_blockedby_pp_q_erlangc'] = \
            ldr_results['prob_blockedby_pp_q_erlangc_results']['model'].predict(
                scenarios_dfs['X_ldr_q'])

        scenarios_io_df['pred_condmeantime_blockedby_pp_onlyq_lm'] = \
            ldr_results['condmeantime_blockedby_pp_onlyq_lm_results']['model'].predict(
                scenarios_dfs['X_ldr_condmeantime_blockedby_pp_onlyq'])

        scenarios_io_df['pred_condmeantime_blockedby_pp_q_mgs'] = \
            ldr_results['condmeantime_blockedby_pp_q_mgc_results']['model'].predict(
                scenarios_dfs['X_ldr_q'])

        print('done with new style predictions for LDR')

        # PP predictions
        features_models_pp = [('basicq', 'lm'), ('noq', 'lm'),
                              ('basicq', 'lassocv'),
                              ('noq', 'poly'), ('basicq', 'poly'),
                              ('noq', 'rf'), ('basicq', 'rf'),
                              ('noq', 'nn'), ('basicq', 'nn')]

        for f, m in features_models_pp:
            scenarios_io_df[f'pred_pp_occ_mean_{f}_{m}'] = \
                pp_results[f'pp_occ_mean_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_pp_{f}'])

            scenarios_io_df[f'pred_pp_occ_p95_{f}_{m}'] = \
                pp_results[f'pp_occ_p95_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_pp_{f}'])

        # onlyq pp
        scenarios_io_df['pred_pp_occ_mean_onlyq_lm'] = \
            pp_results['pp_occ_mean_onlyq_lm_results']['model'].predict(scenarios_dfs['X_pp_occmean_onlyq'])
        scenarios_io_df['pred_pp_occ_p95_onlyq_lm'] = \
            pp_results['pp_occ_p95_onlyq_lm_results']['model'].predict(scenarios_dfs['X_pp_occp95_onlyq'])

        print('done with new style predictions for PP')

        # OBS predictions
        features_models_obs = [('q', 'lm'), ('basicq', 'lm'), ('noq', 'lm'),
                               ('q', 'lassocv'),
                               ('noq', 'poly'), ('basicq', 'poly'),
                               ('noq', 'rf'), ('basicq', 'rf'), ('q', 'rf'),
                               ('noq', 'nn'), ('basicq', 'nn'), ('q', 'nn')]

        for f, m in features_models_obs:
            scenarios_io_df[f'pred_obs_occ_mean_{f}_{m}'] = \
                obs_results[f'obs_occ_mean_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_obs_{f}'])

            scenarios_io_df[f'pred_obs_occ_p95_{f}_{m}'] = \
                obs_results[f'obs_occ_p95_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_obs_{f}'])

            scenarios_io_df[f'pred_prob_blockedby_ldr_{f}_{m}'] = \
                obs_results[f'prob_blockedby_ldr_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_obs_{f}'])

            scenarios_io_df[f'pred_condmeantime_blockedby_ldr_{f}_{m}'] = \
                obs_results[f'condmeantime_blockedby_ldr_{f}_{m}_results']['model'].predict(scenarios_dfs[f'X_obs_{f}'])

        scenarios_io_df['pred_obs_occ_mean_onlyq_lm'] = \
            obs_results['obs_occ_mean_onlyq_lm_results']['model'].predict(scenarios_dfs['X_obs_occmean_onlyq'])

        scenarios_io_df['pred_obs_occ_p95_onlyq_lm'] = \
            obs_results['obs_occ_p95_onlyq_lm_results']['model'].predict(scenarios_dfs['X_obs_occp95_onlyq'])

        scenarios_io_df['pred_prob_blockedby_ldr_onlyq_lm'] = \
            obs_results['prob_blockedby_ldr_onlyq_lm_results']['model'].predict(
                scenarios_dfs['X_obs_prob_blockedby_ldr_onlyq'])

        scenarios_io_df['pred_prob_blockedby_ldr_q_erlangc'] = \
            obs_results['prob_blockedby_ldr_q_erlangc_results']['model'].predict(
                scenarios_dfs['X_obs_q'])

        scenarios_io_df['pred_condmeantime_blockedby_ldr_onlyq_lm'] = \
            obs_results['condmeantime_blockedby_ldr_onlyq_lm_results']['model'].predict(
                scenarios_dfs['X_obs_condmeantime_blockedby_ldr_onlyq'])

        print('done with new style predictions for OBS')

        scenarios_io_df.to_csv(path_scenario_csv, index=True)
        print(f'scenarios_io_df written to {path_scenario_csv}')

        # Create meta inputs scenario file to use for simulation runs
        meta_inputs_df = scenarios_dfs['meta_inputs']
        meta_inputs_path = Path(siminout_path, perf_curve_scenarios_suffix,
                                f'{perf_curve_scenarios_suffix}_obflow06_metainputs_pc.csv')
        meta_inputs_df.to_csv(meta_inputs_path, index=True)
        print(f'meta_inputs_df written to {meta_inputs_path}')
