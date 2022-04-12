import sys
from pathlib import Path
import argparse
import math

import pandas as pd

from obflowsim.mm.obnetwork import prob_blockedby_pp_hat, condmeantime_blockedby_pp_hat, \
    meantime_blockedby_pp_hat, obs_blockedby_ldr_hats
import qng


def create_siminout_qng(siminout_path, siminout_qng_path):

    scenario_siminout_df = pd.read_csv(siminout_path)
    qng_approx_df = qng_approx_from_inputs(scenario_siminout_df)
    scenario_siminout_qng_df = scenario_siminout_df.merge(qng_approx_df, on=['scenario'])
    scenario_siminout_qng_df.to_csv(siminout_qng_path, index=False)


def create_x_y(exp, sim_input_output_qnq_path, output_path):
    """
    Read main data file created by simulation output processing and create X and y dataframes.
    Parameters
    ----------

    exp
    sim_input_output_qnq_path
    output_path

    Returns
    -------

    """

    xy_df = pd.read_csv(sim_input_output_qnq_path, index_col=0)

    # Define which columns are in which matrices starting with no queueing vars
    X_pp_noq_cols = ['arrival_rate', 'mean_los_pp',
                     'c_sect_prob', 'cap_pp']

    X_ldr_noq_cols = ['arrival_rate', 'mean_los_obs', 'cap_obs', 'mean_los_ldr', 'cap_ldr',
                      'mean_los_pp', 'c_sect_prob', 'cap_pp']

    X_obs_noq_cols = ['arrival_rate', 'mean_los_obs', 'cap_obs', 'mean_los_ldr', 'cap_ldr',
                      'mean_los_pp', 'c_sect_prob', 'cap_pp']

    # For "basicq" matrices, only load and rho variables are added
    X_pp_basicq_cols = X_pp_noq_cols.copy()
    X_pp_basicq_cols.extend(['load_pp', 'rho_pp'])

    X_ldr_basicq_cols = X_ldr_noq_cols.copy()
    X_ldr_basicq_cols.extend(['load_obs', 'rho_obs', 'load_ldr',
                              'rho_ldr', 'load_pp', 'rho_pp'])

    X_obs_basicq_cols = X_obs_noq_cols.copy()
    X_obs_basicq_cols.extend(['load_obs', 'rho_obs', 'load_ldr',
                              'rho_ldr', 'load_pp', 'rho_pp'])

    # For "q" matrices, include additional queueing approximations (not applicable
    # to PP since unaffected by upstream unit and has no downstream unit

    # LDR can have LOS shortened by patients blocked in OBS and have LOS lengthened
    # by patients blocked in LDR by PP
    X_ldr_q_cols = X_ldr_basicq_cols.copy()
    X_ldr_q_cols.extend(['probblockedbypp_approx', 'condmeantimeblockedbypp_approx',
                         'probblockedbyldr_approx', 'condmeantimeblockedbyldr_approx',
                         'cv2_svctime_pp',
                         'eff_load_ldr', 'eff_sqrtload_ldr', 'effmean_svctime_ldr_approx'])

    # The onlyq versions can be used to test models based on minimal set of q'ng related variables
    X_pp_occmean_onlyq_cols = ['load_pp']
    X_pp_occp95_onlyq_cols = ['load_pp']

    X_ldr_occmean_onlyq_cols = ['eff_load_ldr']
    X_ldr_occp95_onlyq_cols = ['eff_load_ldr', 'eff_sqrtload_ldr']
    X_ldr_probblocked_onlyq_cols = ['probblockedbypp_approx']
    X_ldr_condmeantimeblocked_onlyq_cols = ['condmeantimeblockedbypp_approx']

    X_obs_occmean_onlyq_cols = ['eff_load_obs']
    X_obs_occp95_onlyq_cols = ['eff_load_obs', 'eff_sqrtload_obs']
    X_obs_probblocked_onlyq_cols = ['probblockedbyldr_approx']
    X_obs_condmeantimeblocked_onlyq_cols = ['condmeantimeblockedbyldr_approx']


    # OBS time in system impacted by
    # congestion in the downstream units.
    X_obs_q_cols = X_obs_basicq_cols.copy()
    X_obs_q_cols.extend(['probblockedbypp_approx', 'condmeantimeblockedbypp_approx',
                         'probblockedbyldr_approx', 'condmeantimeblockedbyldr_approx',
                         'eff_load_obs', 'eff_sqrtload_obs', 'cv2_svctime_obs', 'effmean_svctime_ldr_approx'])

    # Create dataframes based on the column specs above
    X_pp_noq = xy_df.loc[:, X_pp_noq_cols]
    X_ldr_noq = xy_df.loc[:, X_ldr_noq_cols]
    X_obs_noq = xy_df.loc[:, X_obs_noq_cols]

    # PP
    X_pp_basicq = xy_df.loc[:, X_pp_basicq_cols]

    X_pp_occmean_onlyq = xy_df.loc[:, X_pp_occmean_onlyq_cols]
    X_pp_occp95_onlyq = xy_df.loc[:, X_pp_occp95_onlyq_cols]
    X_pp_occp95_onlyq['sqrt_load_pp'] = X_pp_occp95_onlyq['load_pp'] ** 0.5

    # LDR
    X_ldr_basicq = xy_df.loc[:, X_ldr_basicq_cols]
    X_ldr_q = xy_df.loc[:, X_ldr_q_cols]

    X_ldr_occmean_onlyq = xy_df.loc[:, X_ldr_occmean_onlyq_cols]
    X_ldr_occp95_onlyq = xy_df.loc[:, X_ldr_occp95_onlyq_cols]
    X_ldr_probblocked_onlyq = xy_df.loc[:, X_ldr_probblocked_onlyq_cols]
    X_ldr_condmeantimeblocked_onlyq = xy_df.loc[:, X_ldr_condmeantimeblocked_onlyq_cols]

    # OBS
    X_obs_basicq = xy_df.loc[:, X_obs_basicq_cols]
    X_obs_q = xy_df.loc[:, X_obs_q_cols]

    X_obs_occmean_onlyq = xy_df.loc[:, X_obs_occmean_onlyq_cols]
    X_obs_occp95_onlyq = xy_df.loc[:, X_obs_occp95_onlyq_cols]
    X_obs_probblocked_onlyq = xy_df.loc[:, X_obs_probblocked_onlyq_cols]
    X_obs_condmeantimeblocked_onlyq = xy_df.loc[:, X_obs_condmeantimeblocked_onlyq_cols]

    # y vectors
    y_obs_occmean = xy_df.loc[:, 'occ_mean_mean_obs']
    y_obs_occp95 = xy_df.loc[:, 'occ_mean_p95_obs']
    y_ldr_occmean = xy_df.loc[:, 'occ_mean_mean_ldr']
    y_ldr_occp95 = xy_df.loc[:, 'occ_mean_p95_ldr']
    y_pp_occmean = xy_df.loc[:, 'occ_mean_mean_pp']
    y_pp_occp95 = xy_df.loc[:, 'occ_mean_p95_pp']

    y_obs_probblocked = xy_df.loc[:, 'prob_blockedby_ldr']
    y_obs_condmeantimeblocked = xy_df.loc[:, 'condmeantime_blockedby_ldr']

    y_ldr_probblocked = xy_df.loc[:, 'prob_blockedby_pp']
    y_ldr_condmeantimeblocked = xy_df.loc[:, 'condmeantime_blockedby_pp']

    # Write dataframes to csv
    X_pp_noq.to_csv(Path(output_path, f'X_pp_noq_{exp}.csv'))
    X_pp_basicq.to_csv(Path(output_path, f'X_pp_basicq_{exp}.csv'))
    X_pp_occmean_onlyq.to_csv(Path(output_path, f'X_pp_occmean_onlyq_{exp}.csv'))
    X_pp_occp95_onlyq.to_csv(Path(output_path, f'X_pp_occp95_onlyq_{exp}.csv'))

    X_ldr_noq.to_csv(Path(output_path, f'X_ldr_noq_{exp}.csv'))
    X_ldr_basicq.to_csv(Path(output_path, f'X_ldr_basicq_{exp}.csv'))
    X_ldr_q.to_csv(Path(output_path, f'X_ldr_q_{exp}.csv'))
    X_ldr_occmean_onlyq.to_csv(Path(output_path, f'X_ldr_occmean_onlyq_{exp}.csv'))
    X_ldr_occp95_onlyq.to_csv(Path(output_path, f'X_ldr_occp95_onlyq_{exp}.csv'))
    X_ldr_probblocked_onlyq.to_csv(Path(output_path, f'X_ldr_probblocked_onlyq_{exp}.csv'))
    X_ldr_condmeantimeblocked_onlyq.to_csv(Path(output_path, f'X_ldr_condmeantimeblocked_onlyq_{exp}.csv'))

    X_obs_noq.to_csv(Path(output_path, f'X_obs_noq_{exp}.csv'))
    X_obs_basicq.to_csv(Path(output_path, f'X_obs_basicq_{exp}.csv'))
    X_obs_q.to_csv(Path(output_path, f'X_obs_q_{exp}.csv'))
    X_obs_occmean_onlyq.to_csv(Path(output_path, f'X_obs_occmean_onlyq_{exp}.csv'))
    X_obs_occp95_onlyq.to_csv(Path(output_path, f'X_obs_occp95_onlyq_{exp}.csv'))
    X_obs_probblocked_onlyq.to_csv(Path(output_path, f'X_obs_probblocked_onlyq_{exp}.csv'))
    X_obs_condmeantimeblocked_onlyq.to_csv(Path(output_path, f'X_obs_condmeantimeblocked_onlyq_{exp}.csv'))

    y_pp_occmean.to_csv(Path(output_path, f'y_pp_occmean_{exp}.csv'))
    y_pp_occp95.to_csv(Path(output_path, f'y_pp_occp95_{exp}.csv'))
    y_ldr_occmean.to_csv(Path(output_path, f'y_ldr_occmean_{exp}.csv'))
    y_ldr_occp95.to_csv(Path(output_path, f'y_ldr_occp95_{exp}.csv'))
    y_obs_occmean.to_csv(Path(output_path, f'y_obs_occmean_{exp}.csv'))
    y_obs_occp95.to_csv(Path(output_path, f'y_obs_occp95_{exp}.csv'))

    y_ldr_probblocked.to_csv(Path(output_path, f'y_ldr_probblocked_{exp}.csv'))
    y_obs_probblocked.to_csv(Path(output_path, f'y_obs_probblocked_{exp}.csv'))
    y_obs_condmeantimeblocked.to_csv(Path(output_path, f'y_obs_condmeantimeblocked_{exp}.csv'))
    y_ldr_condmeantimeblocked.to_csv(Path(output_path, f'y_ldr_condmeantimeblocked_{exp}.csv'))


def qng_approx_from_inputs(scenario_inputs_summary):
    results = []

    for row in scenario_inputs_summary.iterrows():
        scenario = row[1]['scenario']
        arr_rate = row[1]['arrival_rate']
        c_sect_prob = row[1]['c_sect_prob']
        obs_mean_svctime = row[1]['mean_los_obs']
        obs_cv2_svctime = 1 / row[1]['num_erlang_stages_obs']
        obs_cap = row[1]['cap_obs']
        ldr_mean_svctime = row[1]['mean_los_ldr']
        ldr_cv2_svctime = 1 / row[1]['num_erlang_stages_ldr']
        ldr_cap = row[1]['cap_ldr']
        pp_mean_svctime = c_sect_prob * row[1]['mean_los_pp_c'] + (1 - c_sect_prob) * row[1]['mean_los_pp_noc']
        pp_cap = row[1]['cap_pp']

        rates = [1 / row[1]['mean_los_pp_c'], 1 / row[1]['mean_los_pp_noc']]
        probs = [c_sect_prob, 1 - c_sect_prob]
        stages = [int(row[1]['num_erlang_stages_pp']), int(row[1]['num_erlang_stages_pp'])]
        moments = [qng.qng.hyper_erlang_moment(rates, stages, probs, moment) for moment in [1, 2]]
        variance = moments[1] - moments[0] ** 2
        cv2 = variance / moments[0] ** 2
        pp_cv2_svctime = cv2

        load_obs = arr_rate * obs_mean_svctime
        load_ldr = arr_rate * ldr_mean_svctime
        load_pp = arr_rate * pp_mean_svctime

        sqrt_load_obs = load_obs ** 0.5
        sqrt_load_ldr = load_ldr ** 0.5
        sqrt_load_pp = load_pp ** 0.5
        
        rho_obs = load_obs / obs_cap
        rho_ldr = load_ldr / ldr_cap
        rho_pp = load_pp / pp_cap

        ldr_pct_blockedby_pp = prob_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap,
                                                                         pp_cv2_svctime)
        ldr_meantime_blockedby_pp = condmeantime_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap,
                                                                                      pp_cv2_svctime)
        (obs_meantime_blockedby_ldr, ldr_effmean_svctime, obs_probblockedbyldr, obs_condmeantime_blockedby_ldr) = \
            obs_blockedby_ldr_hats(arr_rate, c_sect_prob, ldr_mean_svctime, ldr_cv2_svctime, ldr_cap,
                                                      pp_mean_svctime, pp_cv2_svctime, pp_cap)

        obs_effmean_svctime = obs_mean_svctime + obs_probblockedbyldr * obs_condmeantime_blockedby_ldr
        eff_load_obs = arr_rate * obs_effmean_svctime
        eff_sqrtload_obs = eff_load_obs ** 0.5
        eff_rho_obs = eff_load_obs / obs_cap

        eff_load_ldr = arr_rate * ldr_effmean_svctime
        eff_sqrtload_ldr = eff_load_ldr ** 0.5
        eff_rho_ldr = eff_load_ldr / ldr_cap

        scen_results = {'scenario': scenario,
                        'mean_los_pp': pp_mean_svctime,
                        'load_obs': load_obs,
                        'load_ldr': load_ldr,
                        'load_pp': load_pp,

                        'sqrt_load_obs': sqrt_load_obs,
                        'sqrt_load_ldr': sqrt_load_ldr,
                        'sqrt_load_pp': sqrt_load_pp,

                        'rho_obs': rho_obs,
                        'rho_ldr': rho_ldr,
                        'rho_pp': rho_pp,
                        'probblockedbyldr_approx': obs_probblockedbyldr,
                        'condmeantimeblockedbyldr_approx': obs_condmeantime_blockedby_ldr,
                        'effmean_svctime_ldr_approx': ldr_effmean_svctime,
                        'probblockedbypp_approx': ldr_pct_blockedby_pp,
                        'condmeantimeblockedbypp_approx': ldr_meantime_blockedby_pp,
                        'cv2_svctime_obs': obs_cv2_svctime,
                        'cv2_svctime_ldr': ldr_cv2_svctime,
                        'cv2_svctime_pp': pp_cv2_svctime,
                        'eff_load_obs': eff_load_obs,
                        'eff_rho_obs': eff_rho_obs,
                        'eff_sqrtload_obs': eff_sqrtload_obs,
                        'eff_load_ldr': eff_load_ldr,
                        'eff_rho_ldr': eff_rho_ldr,
                        'eff_sqrtload_ldr': eff_sqrtload_ldr}

        results.append(scen_results)

    results_df = pd.DataFrame(results)
    return results_df


def process_command_line(argv=None):
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='mm_dataprep',
                                     description='Create X and y files for metamodeling')

    # Add arguments
    parser.add_argument(
        "experiment", type=str,
        help="String used in output filenames"
    )

    parser.add_argument(
        "siminout_qng_path", type=str,
        help="Path to csv file which will contain scenario inputs, summary stats and qng approximations"
    )

    parser.add_argument(
        "output_data_path", type=str,
        help="Path to directory in which to create X and y data files"
    )

    # Do the parsing and return the populated namespace with the input arg values
    args = parser.parse_args()
    return args


def main(argv=None):
    """

    Parameters
    ----------
    argv

    Returns
    -------

    """

    # Parse command line arguments
    args = process_command_line(argv)

    exp = args.experiment
    siminout_qng_path = args.siminout_qng_path
    output_data_path = args.output_data_path

    create_x_y(exp, siminout_qng_path, output_data_path)


if __name__ == '__main__':
    sys.exit(main())


