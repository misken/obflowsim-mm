import sys
import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import t


def compute_occ_stats(obsystem, end_time, warmup=0,
                      quantiles=(0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    """
    Compute occupancy statistics by unit

    Parameters
    ----------
    obsystem : OBsystem
    end_time : float, simulation end time
    egress : bool, if True compute statistics for ENTER and EXIT (default is False)
    log_path : str or Path, location of log files
    warmup : float, simulation warm-up time
    quantiles : tuple, qauntiles to compute

    Returns
    -------

    """
    occ_stats_dfs = []
    occ_dfs = []
    for unit in obsystem.obunits:
        # Only compute if at least onc change in occupancy during simulation
        if len(unit.occupancy_list) > 1:
            occ = unit.occupancy_list

            # Create occupancy dataframe and compute time in each state
            df = pd.DataFrame(occ, columns=['timestamp', 'occ'])
            df['occ_weight'] = -1 * df['timestamp'].diff(periods=-1)

            last_weight = end_time - df.iloc[-1, 0]
            df.fillna(last_weight, inplace=True)
            df['unit'] = unit.name
            occ_dfs.append(df)

            # Filter out recs before warmup
            df = df[df['timestamp'] > warmup]

            # Compute time average occupancy statistics
            weighted_stats = DescrStatsW(df['occ'], weights=df['occ_weight'], ddof=0)

            occ_quantiles = weighted_stats.quantile(quantiles)
            occ_unit_stats_df = pd.DataFrame([{'unit': unit.name, 'capacity': unit.capacity,
                                               'mean_occ': weighted_stats.mean, 'sd_occ': weighted_stats.std,
                                               'min_occ': df['occ'].min(), 'max_occ': df['occ'].max()}])

            # Create wide dataframe of quantiles by unit
            quantiles_df = pd.DataFrame(occ_quantiles).transpose()
            quantiles_df.rename(columns=lambda x: f"p{100 * x:02.0f}_occ", inplace=True)

            occ_unit_stats_df = pd.concat([occ_unit_stats_df, quantiles_df], axis=1)
            occ_stats_dfs.append(occ_unit_stats_df)

    # Combine unit specific occupancy stats
    occ_stats_df = pd.concat(occ_stats_dfs)
    occ_df = pd.concat(occ_dfs)

    return occ_stats_df, occ_df


def num_gt_0(column):
    """
    Helper function to count number of elements > 0 in an array like object
    Parameters
    ----------
    column : pandas Series

    Returns
    -------
    int, number of elements > 0 in ``column``

    """
    return (column > 0).sum()


def get_stats(group, stub=''):
    if group.sum() == 0:
        return {stub + 'count': group.count(), stub + 'mean': 0.0,
                stub + 'min': 0.0, stub + 'num_gt_0': 0,
                stub + 'max': 0.0, 'stdev': 0.0, 'sem': 0.0,
                stub + 'var': 0.0, 'cv': 0.0,
                stub + 'skew': 0.0, 'kurt': 0.0,
                stub + 'p01': 0.0, stub + 'p025': 0.0,
                stub + 'p05': 0.0, stub + 'p25': 0.0,
                stub + 'p50': 0.0, stub + 'p75': 0.0,
                stub + 'p90': 0.0, stub + 'p95': 0.0,
                stub + 'p975': 0.0, stub + 'p99': 0.0}
    else:
        return {stub + 'count': group.count(), stub + 'mean': group.mean(),
                stub + 'min': group.min(), stub + 'num_gt_0': num_gt_0(group),
                stub + 'max': group.max(), 'stdev': group.std(), 'sem': group.sem(),
                stub + 'var': group.var(), 'cv': group.std() / group.mean(),
                stub + 'skew': group.skew(), 'kurt': group.kurt(),
                stub + 'p01': group.quantile(0.01), stub + 'p025': group.quantile(0.025),
                stub + 'p05': group.quantile(0.05), stub + 'p25': group.quantile(0.25),
                stub + 'p50': group.quantile(0.5), stub + 'p75': group.quantile(0.75),
                stub + 'p90': group.quantile(0.9), stub + 'p95': group.quantile(0.95),
                stub + 'p975': group.quantile(0.975), stub + 'p99': group.quantile(0.99)}


def process_obsim_logs(stop_log_path, occ_stats_path, output_path,
                       run_time, warmup=0, output_file_stem='scenario_rep_stats_summary'):
    """
    Creates and writes out summary by scenario and replication to csv

    Parameters
    ----------
    stop_log_path : Path, directory containing stop logs from obflow simulation model
    occ_stats_path : Path, directory containing occupancy stats from obflow simulation model
    output_path : Path, destination for the output summary files
    run_time : float, specified run time for simulation model
    units : tuple of str, unit names for which to compute stats
    warmup : float, time before which data is discarded for computing summary stats
    output_file_stem : str, csv output filename without the extension

    Returns
    -------
    No return value; writes out summary by scenario and replication to csv

    """

    start_analysis = warmup
    end_analysis = run_time
    num_days = (run_time - warmup) / 24.0

    rx = re.compile(r'_scenario_([0-9]{1,4})_rep_([0-9]{1,4})')

    results = []
    active_units = []

    for log_fn in stop_log_path.glob('unit_stop_log*.csv'):
        # Get scenario and rep numbers from filename
        m = re.search(rx, str(log_fn))
        scenario_name = m.group(0)
        scenario_num = int(m.group(1))
        rep_num = int(m.group(2))
        print(scenario_name, scenario_num, rep_num)

        # Read the log file and filter by included categories
        stops_df = pd.read_csv(log_fn)

        stops_df = stops_df[(stops_df['entry_ts'] <= end_analysis) & (stops_df['exit_ts'] >= start_analysis) & \
                   (stops_df['entry_ts'] < stops_df['exit_ts'])]

        # LOS means and sds - planned and actual
        stops_df_grp_unit = stops_df.groupby(['unit'])
        plos_mean = stops_df_grp_unit['planned_los'].mean()
        plos_sd = stops_df_grp_unit['planned_los'].std()
        plos_skew = stops_df_grp_unit['planned_los'].skew()
        plos_kurt = stops_df_grp_unit['planned_los'].apply(pd.DataFrame.kurt)

        actlos_mean = stops_df_grp_unit['exit_enter'].mean()
        actlos_sd = stops_df_grp_unit['exit_enter'].std()
        actlos_skew = stops_df_grp_unit['exit_enter'].skew()
        actlos_kurt = stops_df_grp_unit['exit_enter'].apply(pd.DataFrame.kurt)

        grp_all = stops_df.groupby(['unit'])
        grp_blocked = stops_df[(stops_df['entry_tryentry'] > 0)].groupby(['unit'])

        blocked_uncond_stats = grp_all['entry_tryentry'].apply(get_stats, 'delay_')
        blocked_cond_stats = grp_blocked['entry_tryentry'].apply(get_stats, 'delay_')

        # Create new summary record as dict
        newrec = {'scenario': scenario_num}

        newrec['rep'] = rep_num
        newrec['num_days'] = num_days
        
        # Number of visits to each unit
        units = grp_all.groups.keys()
        for unit in units:
            if (unit, 'delay_count') in blocked_uncond_stats.index:
                newrec[f'num_visits_{unit.lower()}'] = blocked_uncond_stats[(unit, 'delay_count')]
            else:
                newrec[f'num_visits_{unit.lower()}'] = 0

            #newrec[f'num_visits_{unit.lower()}'] = stops_df_grp_unit['exit_ts'].count()[unit]

            # LOS stats for each unit
        for unit in units:
            if newrec[f'num_visits_{unit.lower()}'] > 0:
                active_units.append(unit)

                newrec[f'planned_los_mean_{unit.lower()}'] = plos_mean[unit]
                newrec[f'actual_los_mean_{unit.lower()}'] = actlos_mean[unit]
                newrec[f'planned_los_sd_{unit.lower()}'] = plos_sd[unit]
                newrec[f'actual_los_sd_{unit.lower()}'] = actlos_sd[unit]
        
                newrec[f'planned_los_cv2_{unit.lower()}'] = (plos_sd[unit] / plos_mean[unit]) ** 2
                newrec[f'actual_los_cv2_{unit.lower()}'] = (actlos_sd[unit] / actlos_mean[unit]) ** 2
        
                newrec[f'planned_los_skew_{unit.lower()}'] = plos_skew[unit]
                newrec[f'actual_los_skew_{unit.lower()}'] = actlos_skew[unit]
                newrec[f'planned_los_kurt_{unit.lower()}'] = plos_kurt[unit]
                newrec[f'actual_los_kurt_{unit.lower()}'] = actlos_kurt[unit]

        # Interarrival time stats for each unit
        for unit in units:
            if newrec[f'num_visits_{unit.lower()}'] > 0:
                arrtimes_unit = stops_df.loc[stops_df.unit == unit, 'request_entry_ts']
                # Make sure arrival times are sorted to compute interarrival times
                arrtimes_unit.sort_values(inplace=True)
                iatimes_unit = arrtimes_unit.diff(1)[1:]

                newrec[f'iatime_mean_{unit.lower()}'] = iatimes_unit.mean()
                newrec[f'iatime_sd_{unit.lower()}'] = iatimes_unit.std()
                newrec[f'iatime_skew_{unit.lower()}'] = iatimes_unit.skew()
                newrec[f'iatime_kurt_{unit.lower()}'] = iatimes_unit.kurtosis()

        # Get occ from occ stats summaries
        occ_stats_fn = Path(occ_stats_path) / f"unit_occ_stats_scenario_{scenario_num}_rep_{rep_num}.csv"
        occ_stats_df = pd.read_csv(occ_stats_fn, index_col=0)
        for unit in units:
            if newrec[f'num_visits_{unit.lower()}'] > 0:
                newrec[f'occ_mean_{unit.lower()}'] = occ_stats_df.loc[unit]['mean_occ']
                newrec[f'occ_stdev_{unit.lower()}'] = occ_stats_df.loc[unit]['sd_occ']
                newrec[f'occ_p05_{unit.lower()}'] = occ_stats_df.loc[unit]['p05_occ']
                newrec[f'occ_p25_{unit.lower()}'] = occ_stats_df.loc[unit]['p25_occ']
                newrec[f'occ_p50_{unit.lower()}'] = occ_stats_df.loc[unit]['p50_occ']
                newrec[f'occ_p75_{unit.lower()}'] = occ_stats_df.loc[unit]['p75_occ']
                newrec[f'occ_p95_{unit.lower()}'] = occ_stats_df.loc[unit]['p95_occ']
                newrec[f'occ_p99_{unit.lower()}'] = occ_stats_df.loc[unit]['p99_occ']
                newrec[f'occ_min_{unit.lower()}'] = occ_stats_df.loc[unit]['min_occ']
                newrec[f'occ_max_{unit.lower()}'] = occ_stats_df.loc[unit]['max_occ']

        newrec['prob_blockedby_ldr'] = \
            blocked_uncond_stats[('LDR', 'delay_num_gt_0')] / blocked_uncond_stats[('LDR', 'delay_count')]

        if ('LDR', 'delay_mean') in blocked_cond_stats.index:
            newrec['blockedby_ldr_mean'] = blocked_cond_stats[('LDR', 'delay_mean')]
            newrec['blockedby_ldr_p95'] = blocked_cond_stats[('LDR', 'delay_p95')]
        else:
            newrec['blockedby_ldr_mean'] = 0.0
            newrec['blockedby_ldr_p95'] = 0.0

        newrec['pct_blocked_by_pp'] = \
            blocked_uncond_stats[('PP', 'delay_num_gt_0')] / blocked_uncond_stats[('PP', 'delay_count')]

        if ('PP', 'delay_mean') in blocked_cond_stats.index:
            newrec['blockedby_pp_mean'] = blocked_cond_stats[('PP', 'delay_mean')]
            newrec['blockedby_pp_p95'] = blocked_cond_stats[('PP', 'delay_p95')]
        else:
            newrec['blockedby_pp_mean'] = 0.0
            newrec['blockedby_pp_p95'] = 0.0

        newrec['timestamp'] = str(datetime.now())

        print(newrec)

        results.append(newrec)

        json_stats_path = output_path / 'json'
        json_stats_path.mkdir(exist_ok=True)

        output_json_file = json_stats_path / f'output_stats_scenario_{scenario_num}_rep_{rep_num}.json'
        with open(output_json_file, 'w') as json_output:
            json_output.write(str(newrec) + '\n')

    scenario_rep_summary_df = pd.DataFrame(results)

    output_csv_file = output_path / f'{output_file_stem}.csv'
    scenario_rep_summary_df = scenario_rep_summary_df.sort_values(by=['scenario', 'rep'])
    scenario_rep_summary_df.to_csv(output_csv_file, index=False)
    
    return active_units


def output_header(msg, line_len, scenario, rep_num):
    header = f"\n{msg} (scenario={scenario} rep={rep_num})\n{'-' * line_len}\n"
    return header


def aggregate_over_reps(scen_rep_summary_path, active_units=('OBS', 'LDR', 'CSECT', 'PP')):
    """Compute summary stats by scenario (aggregating over replications)"""

    output_stats_summary_df = pd.read_csv(scen_rep_summary_path).sort_values(by=['scenario', 'rep'])

    unit_dfs = []

    # Need better, more generalizable way to do this, but good enough for now.
    if 'OBS' in active_units:
        output_stats_summary_agg_obs_df = output_stats_summary_df.groupby(['scenario']).agg(
            num_visits_obs_mean=pd.NamedAgg(column='num_visits_obs', aggfunc='mean'),
            planned_los_mean_mean_obs=pd.NamedAgg(column='planned_los_mean_obs', aggfunc='mean'),
            actual_los_mean_mean_obs=pd.NamedAgg(column='actual_los_mean_obs', aggfunc='mean'),
            planned_los_mean_cv2_obs=pd.NamedAgg(column='planned_los_cv2_obs', aggfunc='mean'),
            actual_los_mean_cv2_obs=pd.NamedAgg(column='actual_los_cv2_obs', aggfunc='mean'),
            occ_mean_mean_obs=pd.NamedAgg(column='occ_mean_obs', aggfunc='mean'),
            occ_mean_p95_obs=pd.NamedAgg(column='occ_p95_obs', aggfunc='mean'),
            prob_blockedby_ldr=pd.NamedAgg(column='prob_blockedby_ldr', aggfunc='mean'),
            condmeantime_blockedby_ldr=pd.NamedAgg(column='blockedby_ldr_mean', aggfunc='mean'),
            condp95time_blockedby_ldr=pd.NamedAgg(column='blockedby_ldr_p95', aggfunc='mean')
        )
        unit_dfs.append(output_stats_summary_agg_obs_df)
        
    if 'LDR' in active_units:
        output_stats_summary_agg_ldr_df = output_stats_summary_df.groupby(['scenario']).agg(
            num_visits_ldr_mean=pd.NamedAgg(column='num_visits_ldr', aggfunc='mean'),
            planned_los_mean_mean_ldr=pd.NamedAgg(column='planned_los_mean_ldr', aggfunc='mean'),
            actual_los_mean_mean_ldr=pd.NamedAgg(column='actual_los_mean_ldr', aggfunc='mean'),
            planned_los_mean_cv2_ldr=pd.NamedAgg(column='planned_los_cv2_ldr', aggfunc='mean'),
            actual_los_mean_cv2_ldr=pd.NamedAgg(column='actual_los_cv2_ldr', aggfunc='mean'),
            occ_mean_mean_ldr=pd.NamedAgg(column='occ_mean_ldr', aggfunc='mean'),
            occ_mean_p95_ldr=pd.NamedAgg(column='occ_p95_ldr', aggfunc='mean'),
            prob_blockedby_pp=pd.NamedAgg(column='pct_blocked_by_pp', aggfunc='mean'),
            condmeantime_blockedby_pp=pd.NamedAgg(column='blockedby_pp_mean', aggfunc='mean'),
            condp95time_blockedby_pp=pd.NamedAgg(column='blockedby_pp_p95', aggfunc='mean')
        )
        unit_dfs.append(output_stats_summary_agg_ldr_df)
        
    if 'CSECT' in active_units:
        output_stats_summary_agg_csect_df = output_stats_summary_df.groupby(['scenario']).agg(
            num_visits_csect_mean=pd.NamedAgg(column='num_visits_csect', aggfunc='mean'),
            planned_los_mean_mean_csect=pd.NamedAgg(column='planned_los_mean_csect', aggfunc='mean'),
            actual_los_mean_mean_csect=pd.NamedAgg(column='actual_los_mean_csect', aggfunc='mean'),
            planned_los_mean_cv2_csect=pd.NamedAgg(column='planned_los_cv2_csect', aggfunc='mean'),
            actual_los_mean_cv2_csect=pd.NamedAgg(column='actual_los_cv2_csect', aggfunc='mean'),
            occ_mean_mean_csect=pd.NamedAgg(column='occ_mean_csect', aggfunc='mean'),
            occ_mean_p95_csect=pd.NamedAgg(column='occ_p95_csect', aggfunc='mean')
        )
        unit_dfs.append(output_stats_summary_agg_csect_df)
        
    if 'PP' in active_units:
        output_stats_summary_agg_pp_df = output_stats_summary_df.groupby(['scenario']).agg(
            num_visits_pp_mean=pd.NamedAgg(column='num_visits_pp', aggfunc='mean'),
            planned_los_mean_mean_pp=pd.NamedAgg(column='planned_los_mean_pp', aggfunc='mean'),
            actual_los_mean_mean_pp=pd.NamedAgg(column='actual_los_mean_pp', aggfunc='mean'),
            planned_los_mean_cv2_pp=pd.NamedAgg(column='planned_los_cv2_pp', aggfunc='mean'),
            actual_los_mean_cv2_pp=pd.NamedAgg(column='actual_los_cv2_pp', aggfunc='mean'),
            occ_mean_mean_pp=pd.NamedAgg(column='occ_mean_pp', aggfunc='mean'),
            occ_mean_p95_pp=pd.NamedAgg(column='occ_p95_pp', aggfunc='mean')
        )
        unit_dfs.append(output_stats_summary_agg_pp_df)

    output_stats_summary_agg_df = pd.concat(unit_dfs, axis=1)

    return output_stats_summary_agg_df


def conf_intervals(scenario_rep_summary_df):
    """Compute CIs by scenario (aggregating over replications)"""

    occ_mean_obs_ci = varsum(scenario_rep_summary_df, 'obs', 'occ_mean_obs', 0.05)
    occ_p05_obs_ci = varsum(scenario_rep_summary_df, 'obs', 'occ_p95_obs', 0.05)

    occ_mean_ldr_ci = varsum(scenario_rep_summary_df, 'ldr', 'occ_mean_ldr', 0.05)
    occ_p05_ldr_ci = varsum(scenario_rep_summary_df, 'ldr', 'occ_p95_ldr', 0.05)

    occ_mean_pp_ci = varsum(scenario_rep_summary_df, 'pp', 'occ_mean_pp', 0.05)
    occ_p05_pp_ci = varsum(scenario_rep_summary_df, 'pp', 'occ_p95_pp', 0.05)

    prob_blockedby_ldr_ci = varsum(scenario_rep_summary_df, 'ldr', 'prob_blockedby_ldr', 0.05)
    blockedby_ldr_mean_ci = varsum(scenario_rep_summary_df, 'ldr', 'blockedby_ldr_mean', 0.05)
    blockedby_ldr_p95_ci = varsum(scenario_rep_summary_df, 'ldr', 'blockedby_ldr_p95', 0.05)

    pct_blocked_by_pp_ci = varsum(scenario_rep_summary_df, 'pp', 'pct_blocked_by_pp', 0.05)
    blockedby_pp_mean_ci = varsum(scenario_rep_summary_df, 'pp', 'blockedby_pp_mean', 0.05)
    blockedby_pp_p95_ci = varsum(scenario_rep_summary_df, 'pp', 'blockedby_pp_p95', 0.05)

    ci_dfs = [occ_mean_obs_ci, occ_p05_obs_ci,
              occ_mean_ldr_ci, occ_p05_ldr_ci,
              occ_mean_pp_ci, occ_p05_pp_ci,
              prob_blockedby_ldr_ci, blockedby_ldr_mean_ci, blockedby_ldr_p95_ci,
              pct_blocked_by_pp_ci, blockedby_pp_mean_ci, blockedby_pp_p95_ci]

    ci_df = pd.concat(ci_dfs)
    return ci_df


def create_sim_summaries(scenario_rep_simout_path, output_path, suffix,
                         include_inputs=True,
                         scenario_inputs_path=None,
                         active_units=('OBS', 'LDR', 'CSECT', 'PP')
                         ):
    """

    Parameters
    ----------
    output_path : str or Path, destination for output files
    suffix : str, appended to output filenames
    include_inputs : bool, True to include scenario inputs in output files
    scenario_inputs_path : str or Path, scenario inputs path and filename
    active_units : tuple, units for which to compute summary files

    Returns
    -------
    No return value
    """
    # Create output filename stems
    #scenario_rep_simout_stem_ = f'scenario_rep_simout_{suffix}'
    scenario_simout_stem = f'scenario_simout_{suffix}'
    scenario_siminout_stem = f'scenario_siminout_{suffix}'
    scenario_ci_stem = f'scenario_ci_{suffix}'

    # Compute and output summary stats by scenario (aggregating over the replications)
    scenario_rep_simout_df = pd.read_csv(scenario_rep_simout_path)
    scenario_simout_path = output_path / f"{scenario_simout_stem}.csv"
    scenario_siminout_path = output_path / f"{scenario_siminout_stem}.csv"
    scenario_simout_df = aggregate_over_reps(scenario_rep_simout_path, active_units)
    scenario_simout_df.to_csv(scenario_simout_path, index=True)

    # Compute and output confidence intervals
    scenario_ci_df = conf_intervals(scenario_rep_simout_df)
    scenario_ci_path = output_path / f"{scenario_ci_stem}.csv"
    scenario_ci_df.to_csv(scenario_ci_path, index=True)

    # Merge the scenario summary with the scenario inputs for both aggregated and raw stats
    if include_inputs:

        # Do merge and output for stats by scenario
        scenario_rep_siminout_stem = f'scenario_rep_siminout_{suffix}'
        scenario_simin_df = pd.read_csv(scenario_inputs_path)
        scenario_siminout_df = scenario_simin_df.merge(scenario_simout_df, on=['scenario'])
        scenario_siminout_df.to_csv(scenario_siminout_path, index=False)

        # Do merge and output for stats by scenario, rep
        scenario_rep_siminout_df = scenario_rep_simout_df.merge(scenario_simin_df, on=['scenario'])
        scenario_rep_siminout_path = output_path / f"{scenario_rep_siminout_stem}.csv"
        scenario_rep_siminout_df.to_csv(scenario_rep_siminout_path, index=False)


def varsum(df, unit, pm, alpha):
    """
    Summarize variance in performance measure across replications within a scenario

    Parameters
    ----------
    df : Dataframe, scenario, rep summary
    unit : str
    pm : str, performance measure column name
    alpha : float, used to compute $1-\alpha$ confidence interval

    Returns
    -------
    Dataframe
    """

    pm_varsum_df = df.groupby(['scenario']).agg(
        pm_mean=pd.NamedAgg(column=pm, aggfunc='mean'),
        pm_std=pd.NamedAgg(column=pm, aggfunc='std'),
        pm_n=pd.NamedAgg(column=pm, aggfunc='count'),
        pm_min=pd.NamedAgg(column=pm, aggfunc='min'),
        pm_max=pd.NamedAgg(column=pm, aggfunc='max'))

    # Coefficient of variation
    pm_varsum_df['pm_cv'] = \
        pm_varsum_df['pm_std'] / pm_varsum_df['pm_mean']

    pm_varsum_df['ci_halfwidth'] = \
        t.ppf(1 - alpha / 2, pm_varsum_df['pm_n'] - 1) * pm_varsum_df['pm_std'] / np.sqrt(pm_varsum_df['pm_n'])

    # Precision is CI half width / mean and is measure of relative error
    pm_varsum_df['ci_precision'] = pm_varsum_df['ci_halfwidth'] / pm_varsum_df['pm_mean']

    pm_varsum_df['ci_lower'] = pm_varsum_df['pm_mean'] - pm_varsum_df['ci_halfwidth']
    pm_varsum_df['ci_upper'] = pm_varsum_df['pm_mean'] + pm_varsum_df['ci_halfwidth']

    pm_varsum_df['unit'] = unit
    pm_varsum_df['pm'] = pm

    return pm_varsum_df


def process_stop_log(scenario_num, rep_num, obsystem,  occ_stats_path, run_time, warmup=0):
    """
    Creates and writes out summary by scenario and replication to csv

    Parameters
    ----------


    Returns
    -------
    Dict of summary stats for this scenario and rep

    """

    start_analysis = warmup
    end_analysis = run_time
    num_days = (run_time - warmup) / 24.0

    results = []
    active_units = []

    print(scenario_num, rep_num)

    # Read the log file and filter by included categories
    stops_df = pd.DataFrame(obsystem.stops_timestamps_list)

    stops_df = stops_df[(stops_df['entry_ts'] <= end_analysis) & (stops_df['exit_ts'] >= start_analysis)]

    # LOS means and sds - planned and actual
    stops_df_grp_unit = stops_df.groupby(['unit'])
    plos_mean = stops_df_grp_unit['planned_los'].mean()
    plos_sd = stops_df_grp_unit['planned_los'].std()
    plos_skew = stops_df_grp_unit['planned_los'].skew()
    plos_kurt = stops_df_grp_unit['planned_los'].apply(pd.DataFrame.kurt)

    actlos_mean = stops_df_grp_unit['exit_enter'].mean()
    actlos_sd = stops_df_grp_unit['exit_enter'].std()
    actlos_skew = stops_df_grp_unit['exit_enter'].skew()
    actlos_kurt = stops_df_grp_unit['exit_enter'].apply(pd.DataFrame.kurt)

    grp_all = stops_df.groupby(['unit'])
    grp_blocked = stops_df[(stops_df['entry_tryentry'] > 0)].groupby(['unit'])

    blocked_uncond_stats = grp_all['entry_tryentry'].apply(get_stats, 'delay_')
    blocked_cond_stats = grp_blocked['entry_tryentry'].apply(get_stats, 'delay_')

    # Create new summary record as dict
    newrec = {'scenario': scenario_num}

    newrec['rep'] = rep_num
    newrec['num_days'] = num_days

    # Number of visits to each unit
    units = grp_all.groups.keys()
    for unit in units:
        # if (unit, 'delay_count') in blocked_uncond_stats.index:
        #     newrec[f'num_visits_{unit.lower()}'] = blocked_uncond_stats[(unit, 'delay_count')]
        # else:
        #     newrec[f'num_visits_{unit.lower()}'] = 0

        newrec[f'num_visits_{unit.lower()}'] = stops_df_grp_unit['entry_ts'].count()[unit]

        # LOS stats for each unit
    for unit in units:
        if newrec[f'num_visits_{unit.lower()}'] > 0:
            active_units.append(unit)

            newrec[f'planned_los_mean_{unit.lower()}'] = plos_mean[unit]
            newrec[f'actual_los_mean_{unit.lower()}'] = actlos_mean[unit]
            newrec[f'planned_los_sd_{unit.lower()}'] = plos_sd[unit]
            newrec[f'actual_los_sd_{unit.lower()}'] = actlos_sd[unit]

            if plos_mean[unit] > 0:
                newrec[f'planned_los_cv2_{unit.lower()}'] = (plos_sd[unit] / plos_mean[unit]) ** 2
            else:
                newrec[f'planned_los_cv2_{unit.lower()}'] = 0.0

            if actlos_mean[unit] > 0:
                newrec[f'actual_los_cv2_{unit.lower()}'] = (actlos_sd[unit] / actlos_mean[unit]) ** 2
            else:
                newrec[f'actual_los_cv2_{unit.lower()}'] = 0.0

            newrec[f'planned_los_skew_{unit.lower()}'] = plos_skew[unit]
            newrec[f'actual_los_skew_{unit.lower()}'] = actlos_skew[unit]
            newrec[f'planned_los_kurt_{unit.lower()}'] = plos_kurt[unit]
            newrec[f'actual_los_kurt_{unit.lower()}'] = actlos_kurt[unit]

    # Interarrival time stats for each unit
    for unit in units:
        if newrec[f'num_visits_{unit.lower()}'] > 0:
            arrtimes_unit = stops_df.loc[stops_df.unit == unit, 'request_entry_ts']
            # Make sure arrival times are sorted to compute interarrival times
            arrtimes_unit.sort_values(inplace=True)
            iatimes_unit = arrtimes_unit.diff(1)[1:]

            newrec[f'iatime_mean_{unit.lower()}'] = iatimes_unit.mean()
            newrec[f'iatime_sd_{unit.lower()}'] = iatimes_unit.std()
            newrec[f'iatime_skew_{unit.lower()}'] = iatimes_unit.skew()
            newrec[f'iatime_kurt_{unit.lower()}'] = iatimes_unit.kurtosis()

    # Get occ from occ stats summaries
    #occ_stats_fn = Path(occ_stats_path) / f"unit_occ_stats_scenario_{scenario_num}_rep_{rep_num}.csv"
    occ_stats_df = pd.read_csv(occ_stats_path, index_col=0)
    for unit in units:
        if newrec[f'num_visits_{unit.lower()}'] > 0:
            newrec[f'occ_mean_{unit.lower()}'] = occ_stats_df.loc[unit]['mean_occ']
            newrec[f'occ_stdev_{unit.lower()}'] = occ_stats_df.loc[unit]['sd_occ']
            newrec[f'occ_p05_{unit.lower()}'] = occ_stats_df.loc[unit]['p05_occ']
            newrec[f'occ_p25_{unit.lower()}'] = occ_stats_df.loc[unit]['p25_occ']
            newrec[f'occ_p50_{unit.lower()}'] = occ_stats_df.loc[unit]['p50_occ']
            newrec[f'occ_p75_{unit.lower()}'] = occ_stats_df.loc[unit]['p75_occ']
            newrec[f'occ_p95_{unit.lower()}'] = occ_stats_df.loc[unit]['p95_occ']
            newrec[f'occ_p99_{unit.lower()}'] = occ_stats_df.loc[unit]['p99_occ']
            newrec[f'occ_min_{unit.lower()}'] = occ_stats_df.loc[unit]['min_occ']
            newrec[f'occ_max_{unit.lower()}'] = occ_stats_df.loc[unit]['max_occ']

    newrec['prob_blockedby_ldr'] = \
        blocked_uncond_stats[('LDR', 'delay_num_gt_0')] / blocked_uncond_stats[('LDR', 'delay_count')]

    if ('LDR', 'delay_mean') in blocked_cond_stats.index:
        newrec['blockedby_ldr_mean'] = blocked_cond_stats[('LDR', 'delay_mean')]
        newrec['blockedby_ldr_p95'] = blocked_cond_stats[('LDR', 'delay_p95')]
    else:
        newrec['blockedby_ldr_mean'] = 0.0
        newrec['blockedby_ldr_p95'] = 0.0

    newrec['pct_blocked_by_pp'] = \
        blocked_uncond_stats[('PP', 'delay_num_gt_0')] / blocked_uncond_stats[('PP', 'delay_count')]

    if ('PP', 'delay_mean') in blocked_cond_stats.index:
        newrec['blockedby_pp_mean'] = blocked_cond_stats[('PP', 'delay_mean')]
        newrec['blockedby_pp_p95'] = blocked_cond_stats[('PP', 'delay_p95')]
    else:
        newrec['blockedby_pp_mean'] = 0.0
        newrec['blockedby_pp_p95'] = 0.0

    newrec['timestamp'] = str(datetime.now())

    print(newrec)
    return(newrec)

    # results.append(newrec)
    #
    # json_stats_path = output_path / 'json'
    # json_stats_path.mkdir(exist_ok=True)
    #
    # output_json_file = json_stats_path / f'output_stats_scenario_{scenario_num}_rep_{rep_num}.json'
    # with open(output_json_file, 'w') as json_output:
    #     json_output.write(str(newrec) + '\n')
    #
    # scenario_rep_summary_df = pd.DataFrame(results)
    #
    # output_csv_file = output_path / f'{output_file_stem}.csv'
    # scenario_rep_summary_df = scenario_rep_summary_df.sort_values(by=['scenario', 'rep'])
    # scenario_rep_summary_df.to_csv(output_csv_file, index=False)

    return active_units


def process_command_line(argv=None):
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='obflow_stat',
                                     description='Run inpatient OB simulation output processor')

    # Add arguments
    parser.add_argument(
        "scenario_rep_simout_path", type=str,
        help="Path and filename containing the scenario rep output summary created by obflow_io.concat_stop_summaries"
    )

    parser.add_argument(
        "output_path", type=str,
        help="Destination Path for output summary files"
    )

    parser.add_argument(
        "suffix", type=str,
        help="String to append to various summary filenames"
    )

    parser.add_argument('--process_logs', dest='process_logs', action='store_true')

    parser.add_argument(
        "--stop_log_path", type=str, default=None,
        help="Path containing stop logs"
    )
    parser.add_argument(
        "--occ_stats_path", type=str, default=None,
        help="Path containing occ stats csvs"
    )

    parser.add_argument(
        "--run_time", type=float, default=None,
        help="Simulation run time"
    )

    parser.add_argument(
        "--warmup_time", type=float, default=None,
        help="Simulation warmup time"
    )

    parser.add_argument('--include_inputs', dest='include_inputs', action='store_true')
    parser.add_argument(
        "--scenario_inputs_path", type=str, default=None,
        help="Filename for scenario inputs"
    )

    # do the parsing
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
    args = process_command_line(argv)
    active_units = ('OBS', 'LDR', 'CSECT', 'PP')

    # Optionally process raw output occupancy and stop logs
    if args.process_logs:
        # From the patient stop logs, compute summary stats by scenario by replication
        scenario_rep_summary_stem = f'scenario_rep_simout_{args.suffix}'

        active_units = process_obsim_logs(Path(args.stop_log_path), Path(args.occ_stats_path),
                           Path(args.output_path), args.run_time, warmup=args.warmup_time,
                           output_file_stem=scenario_rep_summary_stem)

    # Create
    create_sim_summaries(Path(args.scenario_rep_simout_path),
                         Path(args.output_path),
                         args.suffix,
                         include_inputs=args.include_inputs,
                         scenario_inputs_path=args.scenario_inputs_path,
                         active_units=active_units)


if __name__ == '__main__':
    sys.exit(main())

