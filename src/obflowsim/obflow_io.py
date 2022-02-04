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


def write_stop_log(csv_path, obsystem, egress=False):
    timestamp_df = pd.DataFrame(obsystem.stops_timestamps_list)
    if egress:
        timestamp_df.to_csv(csv_path, index=False)
    else:
        timestamp_df[(timestamp_df['unit'] != 'ENTRY') &
                     (timestamp_df['unit'] != 'EXIT')].to_csv(csv_path, index=False)

    if egress:
        timestamp_df.to_csv(csv_path, index=False)
    else:
        timestamp_df[(timestamp_df['unit'] != 'ENTRY') &
                     (timestamp_df['unit'] != 'EXIT')].to_csv(csv_path, index=False)


def output_header(msg, linelen, scenario, rep_num):
    header = f"\n{msg} (scenario={scenario} rep={rep_num})\n{'-' * linelen}\n"
    return header


