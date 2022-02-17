from pathlib import Path
import argparse
import math

import pandas as pd

from .obnetwork import prob_blockedby_pp_hat, condmeantime_blockedby_pp_hat, obs_blockedby_ldr_hats
import qng


def create_siminout_qng(siminout_path, siminout_qng_path):

    scenario_siminout_df = pd.read_csv(siminout_path)
    qng_approx_df = qng_approx_from_inputs(scenario_siminout_df)
    scenario_siminout_qng_df = scenario_siminout_df.merge(qng_approx_df, on=['scenario'])
    scenario_siminout_qng_df.to_csv(siminout_qng_path, index=False)


def create_x_y(exp, sim_input_output_qnq_path, scenarios, output_path):
    """
    Read main data file created by simulation output processing and create X and y dataframes.
    Parameters
    ----------
    scenarios
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

    X_ldr_noq_cols = ['arrival_rate', 'mean_los_obs', 'mean_los_ldr', 'cap_ldr',
                      'mean_los_pp', 'c_sect_prob', 'cap_pp']

    X_obs_noq_cols = ['arrival_rate', 'mean_los_obs', 'cap_obs', 'mean_los_ldr', 'cap_ldr',
                      'mean_los_pp', 'c_sect_prob', 'cap_pp']

    # For "basicq" matrices, only load and rho variables are added
    X_pp_basicq_cols = X_pp_noq_cols.copy()
    X_pp_basicq_cols.extend(['load_pp', 'rho_pp'])

    X_ldr_basicq_cols = X_ldr_noq_cols.copy()
    X_ldr_basicq_cols.extend(['load_ldr', 'rho_ldr', 'load_pp', 'rho_pp'])

    X_obs_basicq_cols = X_obs_noq_cols.copy()
    X_obs_basicq_cols.extend(['load_obs', 'rho_obs', 'load_ldr',
                              'rho_ldr', 'load_pp', 'rho_pp'])

    # For "q" matrices, include additional queueing approximations (not applicable
    # to PP since unaffected by upstream unit and has no downstream unit

    # LDR can have LOS shortened by patients blocked in OBS and have LOS lengthened
    # by patients blocked in LDR by PP
    X_ldr_q_cols = X_ldr_basicq_cols.copy()
    X_ldr_q_cols.extend(['prob_blockedby_pp_approx', 'condmeantime_blockedby_pp_approx',
                         'prob_blockedby_ldr_approx', 'condmeantime_blockedby_ldr_approx',
                         'pp_cv2_svctime',
                         'ldr_eff_load', 'ldr_eff_sqrtload', 'ldr_effmean_svctime_approx'])

    # The onlyq versions can be used to test models based on minimal set of q'ng related variables
    X_pp_occmean_onlyq_cols = ['load_pp']
    X_pp_occp95_onlyq_cols = ['load_pp']

    X_ldr_occmean_onlyq_cols = ['ldr_eff_load']
    X_ldr_occp95_onlyq_cols = ['ldr_eff_load', 'ldr_eff_sqrtload']
    X_ldr_prob_blockedby_pp_onlyq_cols = ['prob_blockedby_pp_approx']
    X_ldr_condmeantime_blockedby_pp_onlyq_cols = ['condmeantime_blockedby_pp_approx']

    X_obs_occmean_onlyq_cols = ['obs_eff_load']
    X_obs_occp95_onlyq_cols = ['obs_eff_load', 'obs_eff_sqrtload']
    X_obs_prob_blockedby_ldr_onlyq_cols = ['prob_blockedby_ldr_approx']
    X_obs_condmeantime_blockedby_ldr_onlyq_cols = ['condmeantime_blockedby_ldr_approx']


    # OBS modeled as infinite capacity system but time in system impacted by
    # congestion in the downstream units.
    X_obs_q_cols = X_obs_basicq_cols.copy()
    X_obs_q_cols.extend(['prob_blockedby_pp_approx', 'condmeantime_blockedby_pp_approx',
                         'prob_blockedby_ldr_approx', 'condmeantime_blockedby_ldr_approx',
                         'ldr_cv2_svctime',
                         'obs_eff_load', 'obs_eff_sqrtload', 'ldr_effmean_svctime_approx'])

    # Create dataframes based on the column specs above
    X_pp_noq = xy_df.loc[scenarios, X_pp_noq_cols]
    X_ldr_noq = xy_df.loc[scenarios, X_ldr_noq_cols]
    X_obs_noq = xy_df.loc[scenarios, X_obs_noq_cols]

    # PP
    X_pp_basicq = xy_df.loc[scenarios, X_pp_basicq_cols]
    X_pp_basicq['sqrt_load_pp'] = X_pp_basicq['load_pp'] ** 0.5

    X_pp_occmean_onlyq = xy_df.loc[scenarios, X_pp_occmean_onlyq_cols]
    X_pp_occp95_onlyq = xy_df.loc[scenarios, X_pp_occp95_onlyq_cols]
    X_pp_occp95_onlyq['sqrt_load_pp'] = X_pp_occp95_onlyq['load_pp'] ** 0.5

    # LDR
    X_ldr_basicq = xy_df.loc[scenarios, X_ldr_basicq_cols]
    X_ldr_basicq['sqrt_load_ldr'] = X_ldr_basicq['load_ldr'] ** 0.5
    X_ldr_basicq['sqrt_load_pp'] = X_ldr_basicq['load_pp'] ** 0.5

    X_ldr_q = xy_df.loc[scenarios, X_ldr_q_cols]
    X_ldr_q['sqrt_load_ldr'] = X_ldr_q['load_ldr'] ** 0.5
    X_ldr_q['sqrt_load_pp'] = X_ldr_q['load_pp'] ** 0.5

    X_ldr_occmean_onlyq = xy_df.loc[scenarios, X_ldr_occmean_onlyq_cols]
    X_ldr_occp95_onlyq = xy_df.loc[scenarios, X_ldr_occp95_onlyq_cols]
    X_ldr_prob_blockedby_pp_onlyq = xy_df.loc[scenarios, X_ldr_prob_blockedby_pp_onlyq_cols]
    X_ldr_condmeantime_blockedby_pp_onlyq = xy_df.loc[scenarios, X_ldr_condmeantime_blockedby_pp_onlyq_cols]

    # OBS
    X_obs_basicq = xy_df.loc[scenarios, X_obs_basicq_cols]
    X_obs_basicq['sqrt_load_obs'] = X_obs_basicq['load_obs'] ** 0.5
    X_obs_basicq['sqrt_load_ldr'] = X_obs_basicq['load_ldr'] ** 0.5
    X_obs_basicq['sqrt_load_pp'] = X_obs_basicq['load_pp'] ** 0.5

    X_obs_q = xy_df.loc[scenarios, X_obs_q_cols]
    X_obs_q['sqrt_load_obs'] = X_obs_q['load_obs'] ** 0.5
    X_obs_q['sqrt_load_ldr'] = X_obs_q['load_ldr'] ** 0.5
    X_obs_q['sqrt_load_pp'] = X_obs_q['load_pp'] ** 0.5

    X_obs_occmean_onlyq = xy_df.loc[scenarios, X_obs_occmean_onlyq_cols]
    X_obs_occp95_onlyq = xy_df.loc[scenarios, X_obs_occp95_onlyq_cols]
    X_obs_prob_blockedby_ldr_onlyq = xy_df.loc[scenarios, X_obs_prob_blockedby_ldr_onlyq_cols]
    X_obs_condmeantime_blockedby_ldr_onlyq = xy_df.loc[scenarios, X_obs_condmeantime_blockedby_ldr_onlyq_cols]

    # y vectors
    y_pp_occ_mean = xy_df.loc[scenarios, 'occ_mean_mean_pp']
    y_pp_occ_p95 = xy_df.loc[scenarios, 'occ_mean_p95_pp']
    y_ldr_occ_mean = xy_df.loc[scenarios, 'occ_mean_mean_ldr']
    y_ldr_occ_p95 = xy_df.loc[scenarios, 'occ_mean_p95_ldr']
    y_obs_occ_mean = xy_df.loc[scenarios, 'occ_mean_mean_obs']
    y_obs_occ_p95 = xy_df.loc[scenarios, 'occ_mean_p95_obs']

    y_prob_blockedby_pp = xy_df.loc[scenarios, 'prob_blockedby_pp']
    y_prob_blockedby_ldr = xy_df.loc[scenarios, 'prob_blockedby_ldr']
    y_condmeantime_blockedby_ldr = xy_df.loc[scenarios, 'condmeantime_blockedby_ldr']
    y_condmeantime_blockedby_pp = xy_df.loc[scenarios, 'condmeantime_blockedby_pp']

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
    X_ldr_prob_blockedby_pp_onlyq.to_csv(Path(output_path, f'X_ldr_prob_blockedby_pp_onlyq_{exp}.csv'))
    X_ldr_condmeantime_blockedby_pp_onlyq.to_csv(Path(output_path, f'X_ldr_condmeantime_blockedby_pp_onlyq_{exp}.csv'))

    X_obs_noq.to_csv(Path(output_path, f'X_obs_noq_{exp}.csv'))
    X_obs_basicq.to_csv(Path(output_path, f'X_obs_basicq_{exp}.csv'))
    X_obs_q.to_csv(Path(output_path, f'X_obs_q_{exp}.csv'))
    X_obs_occmean_onlyq.to_csv(Path(output_path, f'X_obs_occmean_onlyq_{exp}.csv'))
    X_obs_occp95_onlyq.to_csv(Path(output_path, f'X_obs_occp95_onlyq_{exp}.csv'))
    X_obs_prob_blockedby_ldr_onlyq.to_csv(Path(output_path, f'X_obs_prob_blockedby_ldr_onlyq_{exp}.csv'))
    X_obs_condmeantime_blockedby_ldr_onlyq.to_csv(Path(output_path, f'X_obs_condmeantime_blockedby_ldr_onlyq_{exp}.csv'))

    y_pp_occ_mean.to_csv(Path(output_path, f'y_pp_occ_mean_{exp}.csv'))
    y_pp_occ_p95.to_csv(Path(output_path, f'y_pp_occ_p95_{exp}.csv'))
    y_ldr_occ_mean.to_csv(Path(output_path, f'y_ldr_occ_mean_{exp}.csv'))
    y_ldr_occ_p95.to_csv(Path(output_path, f'y_ldr_occ_p95_{exp}.csv'))
    y_obs_occ_mean.to_csv(Path(output_path, f'y_obs_occ_mean_{exp}.csv'))
    y_obs_occ_p95.to_csv(Path(output_path, f'y_obs_occ_p95_{exp}.csv'))

    y_prob_blockedby_pp.to_csv(Path(output_path, f'y_prob_blockedby_pp_{exp}.csv'))
    y_prob_blockedby_ldr.to_csv(Path(output_path, f'y_prob_blockedby_ldr_{exp}.csv'))
    y_condmeantime_blockedby_ldr.to_csv(Path(output_path, f'y_condmeantime_blockedby_ldr_{exp}.csv'))
    y_condmeantime_blockedby_pp.to_csv(Path(output_path, f'y_condmeantime_blockedby_pp_{exp}.csv'))


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
        
        rho_obs = load_obs / obs_cap
        rho_ldr = load_ldr / ldr_cap
        rho_pp = load_pp / pp_cap

        ldr_pct_blockedby_pp = prob_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap,
                                                                         pp_cv2_svctime)
        ldr_meantime_blockedby_pp = condmeantime_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap,
                                                                                      pp_cv2_svctime)
        (obs_meantime_blockedby_ldr, ldr_effmean_svctime, obs_prob_blockedby_ldr, obs_condmeantime_blockedby_ldr) = \
            obs_blockedby_ldr_hats(arr_rate, c_sect_prob, ldr_mean_svctime, ldr_cv2_svctime, ldr_cap,
                                                      pp_mean_svctime, pp_cv2_svctime, pp_cap)

        obs_effmean_svctime = obs_mean_svctime + obs_prob_blockedby_ldr * obs_condmeantime_blockedby_ldr
        obs_eff_load = arr_rate * obs_effmean_svctime
        obs_eff_sqrtload = obs_eff_load ** 0.5
        obs_eff_rho = obs_eff_load / obs_cap

        ldr_eff_load = arr_rate * ldr_effmean_svctime
        ldr_eff_sqrtload = ldr_eff_load ** 0.5
        ldr_eff_rho = ldr_eff_load / ldr_cap

        scen_results = {'scenario': scenario,
                        'mean_los_pp': pp_mean_svctime,
                        'load_obs': load_obs,
                        'load_ldr': load_ldr,
                        'load_pp': load_pp,
                        'rho_obs': rho_obs,
                        'rho_ldr': rho_ldr,
                        'rho_pp': rho_pp,
                        'prob_blockedby_ldr_approx': obs_prob_blockedby_ldr,
                        'condmeantime_blockedby_ldr_approx': obs_condmeantime_blockedby_ldr,
                        'ldr_effmean_svctime_approx': ldr_effmean_svctime,
                        'prob_blockedby_pp_approx': ldr_pct_blockedby_pp,
                        'condmeantime_blockedby_pp_approx': ldr_meantime_blockedby_pp,
                        'obs_cv2_svctime': obs_cv2_svctime,
                        'ldr_cv2_svctime': ldr_cv2_svctime,
                        'pp_cv2_svctime': pp_cv2_svctime,
                        'obs_eff_load': obs_eff_load,
                        'obs_eff_rho': obs_eff_rho,
                        'obs_eff_sqrtload': obs_eff_sqrtload,
                        'ldr_eff_load': ldr_eff_load,
                        'ldr_eff_rho': ldr_eff_rho,
                        'ldr_eff_sqrtload': ldr_eff_sqrtload}

        results.append(scen_results)

    results_df = pd.DataFrame(results)
    return results_df


def process_command_line():
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='dataprep',
                                     description='Create X and y files for metamodeling')

    # Add arguments
    parser.add_argument(
        "experiment", type=str,
        help="String used in output filenames"
    )

    parser.add_argument(
        "siminout_path", type=str,
        help="Path to csv file containing scenario inputs, summary stats"
    )


    parser.add_argument(
        "siminout_qng_path", type=str,
        help="Path to csv file which will contain scenario inputs, summary stats and qng approximations"
    )

    parser.add_argument(
        "output_data_path", type=str,
        help="Path to directory in which to create X and y data files"
    )

    parser.add_argument(
        "scenario_start", type=str, default=None,
        help="Start of slice object for use in pandas loc selector"
    )

    parser.add_argument(
        "scenario_end", type=str, default=None,
        help="End of slice object for use in pandas loc selector"
    )

    # Do the parsing and return the populated namespace with the input arg values
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    override_args = True

    if override_args:
        exp = "exp12"
        siminout_path = Path("data", "siminout", f"scenario_siminout_{exp}.csv")
        siminout_qng_path = Path("data", "siminout", f"scenario_siminout_qng_{exp}.csv")
        output_data_path = Path("data")
        scenarios = slice(1, 2880)
    else:
        mm_args = process_command_line()
        exp = mm_args.experiment
        siminout_path = mm_args.siminout_path
        siminout_qng_path = mm_args.siminout_qng_path
        if (mm_args.scenario_start is not None) and (mm_args.scenario_end is not None):
            scenarios = slice(int(mm_args.scenario_start), int(mm_args.scenario_end))
        else:
            scenarios = None
        output_data_path = mm_args.output_data_path

    create_siminout_qng(siminout_path, siminout_qng_path)
    create_x_y(exp, siminout_qng_path, scenarios, output_data_path)
