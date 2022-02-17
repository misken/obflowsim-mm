import sys
import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(cfg):
    """

    Parameters
    ----------
    cfg : str, configuration filename

    Returns
    -------
    dict
    """

    # Read inputs from config file
    with open(cfg, 'rt') as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)

    return yaml_config


def write_stop_log(csv_path, obsystem, egress=True):
    """

    Parameters
    ----------
    csv_path
    obsystem
    egress

    Returns
    -------

    """
    stops_df = pd.DataFrame(obsystem.stops_timestamps_list)
    if egress:
        stops_df.to_csv(csv_path, index=False)
    else:
        stops_df[(stops_df['unit'] != 'ENTRY') &
                     (stops_df['unit'] != 'EXIT')].to_csv(csv_path, index=False)

    if egress:
        stops_df.to_csv(csv_path, index=False)
    else:
        stops_df[(stops_df['unit'] != 'ENTRY') &
                     (stops_df['unit'] != 'EXIT')].to_csv(csv_path, index=False)


def concat_stop_summaries(stop_summaries_path, output_path,
                          summary_stats_file_stem='summary_stats_scenario',
                          output_file_stem=f'scenario_rep_simout'):
    """
    Creates and writes out summary by scenario and replication to csv

    Parameters
    ----------
    stop_summaries_path
    output_path
    summary_stats_file_stem
    output_file_stem

    Returns
    -------

    """

    summary_files = [fn for fn in Path(stop_summaries_path).glob(f'{summary_stats_file_stem}*.csv')]
    scenario_rep_summary_df = pd.concat([pd.read_csv(fn) for fn in summary_files])

    output_csv_file = Path(output_path) / f'{output_file_stem}.csv'
    scenario_rep_summary_df = scenario_rep_summary_df.sort_values(by=['scenario', 'rep'])

    scenario_rep_summary_df.to_csv(output_csv_file, index=False)


def write_occ_log(csv_path, occ_df, egress=False):
    """
    Export raw occupancy logs to csv

    Parameters
    ----------
    csv_path
    occ_df
    egress

    Returns
    -------

    """

    if egress:
        occ_df.to_csv(csv_path, index=False)
    else:
        occ_df[(occ_df['unit'] != 'ENTRY') &
               (occ_df['unit'] != 'EXIT')].to_csv(csv_path, index=False)


def write_occ_stats(occ_stats_path, occ_stats_df):
    """
    Export occupancy stats to csv

    Parameters
    ----------
    occ_stats_path
    occ_stats_df

    Returns
    -------
    """

    occ_stats_df.to_csv(occ_stats_path, index=False)


def write_summary_stats(summary_stats_path, summary_stats_df):
    """
    Export occupancy stats to csv

    Parameters
    ----------
    summary_stats_path
    summary_stats_df

    Returns
    -------
    """

    summary_stats_df.to_csv(summary_stats_path, index=False)


def output_header(msg, linelen, scenario, rep_num):

    header = f"\n{msg} (scenario={scenario} rep={rep_num})\n{'-' * linelen}\n"
    return header


def process_command_line(argv=None):
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='obflow_6',
                                     description='Run inpatient OB simulation')

    # Add arguments
    parser.add_argument(
        "stop_summaries_path", type=str,
        help="Folder containing the scenario rep summaries created by simulation runs"
    )

    parser.add_argument(
        "output_path", type=str,
        help="Destination folder for combined scenario rep summary csv"
    )

    parser.add_argument(
        "summary_stats_file_stem", type=str,
        help="Summary stat file name without extension"
    )

    parser.add_argument(
        "output_file_stem", type=str,
        help="Combined summary stat file name without extension to be output"
    )

    # do the parsing
    args = parser.parse_args(argv)

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

    concat_stop_summaries(args.stop_summaries_path,
                          args.output_path,
                          args.summary_stats_file_stem,
                          args.output_file_stem)


if __name__ == '__main__':
    sys.exit(main())
