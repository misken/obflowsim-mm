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


