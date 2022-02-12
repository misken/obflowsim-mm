import numpy as np
import pandas as pd
from pathlib import Path
import itertools
import json
import yaml

from qng import qng
from mm.mm_dataprep import qng_approx_from_inputs


def scenario_grid_to_csv(path_scenario_grid_yaml, _meta_inputs_path):
    """
    Creates obsimpy metainputs csv file from scenario grid YAML file

    Parameters
    ----------
    path_scenario_grid_yaml
    _meta_inputs_path

    Returns
    -------
    None. The metainputs csv file is written to ``meta_inputs_path``.
    """

    with open(path_scenario_grid_yaml, "r") as stream:
        try:
            _scenario_grid = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    input_scenarios = [scn for scn in itertools.product(*[value for key, value in _scenario_grid.items()])]
    cols = list(_scenario_grid.keys())
    scenario_simin_df = pd.DataFrame(input_scenarios, columns=cols)

    # Compute capacity using inverse Poisson and accommodation targets
    scenario_simin_df['cap_obs'] = \
        scenario_simin_df.apply(lambda x: qng.poissoninv(x.acc_obs, x.arrival_rate * x.mean_los_obs), axis=1)
    scenario_simin_df['cap_ldr'] = \
        scenario_simin_df.apply(lambda x: qng.poissoninv(x.acc_ldr, x.arrival_rate * x.mean_los_ldr), axis=1)
    scenario_simin_df['cap_pp'] = \
        scenario_simin_df.apply(
            lambda x: qng.poissoninv(x.acc_pp,
                                     x.arrival_rate * (x.c_sect_prob * x.mean_los_pp_c + (1 - x.c_sect_prob) * x.mean_los_pp_noc)), axis=1)

    # Compute check values for load and rho based on computed cap levels
    # These values WILL get recomputed when creating simulation input config files based on
    # whatever capacity numbers (cap_obs, cap_ldr, cap_pp) are used in the scenarios metainput file.

    scenario_simin_df['check_load_obs'] = \
        scenario_simin_df.apply(lambda x: x.arrival_rate * x.mean_los_obs, axis=1)
    scenario_simin_df['check_load_ldr'] = \
        scenario_simin_df.apply(lambda x: x.arrival_rate * x.mean_los_ldr, axis=1)
    scenario_simin_df['check_load_pp'] = \
        scenario_simin_df.apply(
            lambda x:  x.arrival_rate * (x.c_sect_prob * x.mean_los_pp_c + (1 - x.c_sect_prob) * x.mean_los_pp_noc),
            axis=1)

    scenario_simin_df['check_rho_obs'] = \
        scenario_simin_df.apply(lambda x: round(x.check_load_obs / x.cap_obs, 2), axis=1)
    scenario_simin_df['check_rho_ldr'] = \
        scenario_simin_df.apply(lambda x: round(x.check_load_ldr / x.cap_ldr, 2), axis=1)
    scenario_simin_df['check_rho_pp'] = \
        scenario_simin_df.apply(lambda x: round(x.check_load_pp / x.cap_pp, 2), axis=1)


    num_scenarios = len(scenario_simin_df.index)
    scenario_simin_df['scenario'] = np.arange(1, num_scenarios + 1)
    #scenario_simin_df.set_index(np.arange(1, num_scenarios + 1), inplace=True)
    #scenario_simin_df.index.name = 'scenario'

    qng_approx_df = qng_approx_from_inputs(scenario_simin_df)
    scenario_simin_qng_df = scenario_simin_df.merge(qng_approx_df, on=['scenario'])
    scenario_simin_qng_df.to_csv(_meta_inputs_path, index=False)


    # Create meta inputs scenario file to use for simulation runs
    # scenario_simin_df.to_csv(_meta_inputs_path, index=True)
    print(f'Metainputs csv file written to {_meta_inputs_path}')


# The following inputs need to be in CLI

output_path = Path("input") # Destination for YAML scenarios file
exp = 'exp13' # Used to create subdirs and filenames
siminout_path = Path("input") # Destination for metainputs csv file based on scenarios

# Need to come up with way to make this dict a file driven thing (e.g. JSON or YAML)
scenario_grid = {'arrival_rate': np.linspace(0.2, 1.0, num=5), 'mean_los_obs': np.array([1.0, 2.0, 5.0]),
                 'mean_los_ldr': np.array([12.0]), 'mean_los_csect': np.atleast_1d(2.0),
                 'mean_los_pp_noc': np.array([24.0, 48.0]), 'mean_los_pp_c': np.array([48.0, 72.0]),
                 'c_sect_prob': np.array([0.15, 0.25, 0.35]),
                 'num_erlang_stages_obs': np.atleast_1d(1), 'num_erlang_stages_ldr': np.atleast_1d(2),
                 'num_erlang_stages_csect': np.atleast_1d(1), 'num_erlang_stages_pp': np.atleast_1d(8),
                 'acc_obs': np.array([0.95, 0.99]), 'acc_ldr': np.array([0.85, 0.9, 0.99]), 'acc_pp': np.array([0.85, 0.9, 0.99])}

# Make scenario specific subdirectory if it doesn't already exist for writing the meta
# inputs file to
Path(siminout_path, exp, ).mkdir(exist_ok=True)
meta_inputs_path = Path(siminout_path, exp, f'{exp}_obflowsim_metainputs.csv')

# Create scenario lists from the grid specs above
scenario_grid_lists = {key: value.tolist() for key, value in scenario_grid.items()}

json_file_path = Path(output_path, f'scenario_grid_{exp}.json')
yaml_file_path = Path(output_path, f'scenario_grid_{exp}.yaml')

with open(json_file_path, 'w') as f_json:
    json.dump(scenario_grid_lists, f_json, sort_keys=False, indent=4)

with open(yaml_file_path, 'w') as f_yaml:
    yaml.dump(scenario_grid_lists, f_yaml, sort_keys=False)
    print(f'Scenario grid YAML file written to {yaml_file_path}')

scenario_grid_to_csv(yaml_file_path, meta_inputs_path)
