import numpy as np

from scenario_tools import create_scenario_recipe_yaml

# Need to come up with way to make this dict a file driven thing (e.g. JSON or YAML)
    # Just represent values as lists in YAML file
scenario_grid = {'arrival_rate': np.linspace(0.2, 1.0, num=3), 'mean_los_obs': np.array([2.0]),
                 'mean_los_ldr': np.array([12.0]), 'mean_los_csect': np.atleast_1d(2.0),
                 'mean_los_pp_noc': np.array([48.0]), 'mean_los_pp_c': np.array([72.0]),
                 'c_sect_prob': np.array([0.25]),
                 'num_erlang_stages_obs': np.atleast_1d(1), 'num_erlang_stages_ldr': np.atleast_1d(2),
                 'num_erlang_stages_csect': np.atleast_1d(1), 'num_erlang_stages_pp': np.atleast_1d(8),
                 'acc_obs': np.array([0.95, 0.99]), 'acc_ldr': np.array([0.85, 0.95]),
                 'acc_pp': np.array([0.85, 0.95])}

exp = 'exp14'
yaml_output_file_path = f'./input/{exp}/{exp}_scenario_recipe.yaml'
create_scenario_recipe_yaml(scenario_grid, yaml_output_file_path)