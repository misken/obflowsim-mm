import argparse
import sys

import numpy as np
import pandas as pd
from pathlib import Path
import itertools
import json
import yaml

from qng import qng
from obflowsim.mm.mm_dataprep import qng_approx_from_inputs


def scenario_grid_to_csv(scenario_recipe_yaml_file, _csv_output_file_path):
    """
    Creates obsimpy metainputs csv file from scenario grid YAML file.

    Parameters
    ----------
    scenario_recipe_yaml_file : str, YAML filename for scenario grid file
    _csv_output_file_path : str or Path, filename for output csv file

    Returns
    -------
    None. The metainputs csv file is written to ``meta_inputs_path``.
    """

    with open(scenario_recipe_yaml_file, "r") as stream:
        try:
            scenario_grid = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    input_scenarios = [scn for scn in itertools.product(*[value for key, value in scenario_grid.items()])]
    cols = list(scenario_grid.keys())
    scenario_simin_df = pd.DataFrame(input_scenarios, columns=cols)

    # Compute capacity lower bound using inverse Poisson and accommodation targets
    scenario_simin_df['cap0_obs'] = \
        scenario_simin_df.apply(lambda x: qng.poissoninv(x.acc_obs, x.arrival_rate * x.mean_los_obs), axis=1)
    scenario_simin_df['cap0_ldr'] = \
        scenario_simin_df.apply(lambda x: qng.poissoninv(x.acc_ldr, x.arrival_rate * x.mean_los_ldr), axis=1)
    scenario_simin_df['cap0_pp'] = \
        scenario_simin_df.apply(
            lambda x: qng.poissoninv(x.acc_pp,
                                     x.arrival_rate * (x.c_sect_prob * x.mean_los_pp_c + (1 - x.c_sect_prob) * x.mean_los_pp_noc)), axis=1)

    # Set actual capacity to lower bound (cap0) if capacity levels not specified in grid
    if 'cap_obs' not in cols:
        scenario_simin_df['cap_obs'] = scenario_simin_df['cap0_obs']

    if 'cap_ldr' not in cols:
        scenario_simin_df['cap_ldr'] = scenario_simin_df['cap0_ldr']

    if 'cap_pp' not in cols:
        scenario_simin_df['cap_pp'] = scenario_simin_df['cap0_pp']


    # Compute check values for load and rho based on computed cap levels
    # These values can get recomputed when creating simulation input config files based on
    # whatever capacity numbers (cap_obs, cap_ldr, cap_pp) are used in the scenarios input file. User
    # can indicate whether or not to update these values - no need to do so if capacity numbers not changed from
    # those set via the Poisson inverse computations.

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

    # Adding the queueing approximation terms so input file can be used for metamodeling
    #  Also nice to have approximations to compare to simulation outputs for validation/debugging
    qng_approx_df = qng_approx_from_inputs(scenario_simin_df)
    scenario_simin_qng_df = scenario_simin_df.merge(qng_approx_df, on=['scenario'])

    'Move scenario to 1st column'
    cols = scenario_simin_qng_df.columns.tolist()
    new_cols = ['scenario']
    other_cols = [c for c in cols if c != 'scenario']
    new_cols.extend(other_cols)
    scenario_simin_qng_df = scenario_simin_qng_df[new_cols]

    # Create sim inputs scenario file to use for simulation runs
    scenario_simin_qng_df.to_csv(_csv_output_file_path, index=False)
    print(f'Metainputs csv file written to {_csv_output_file_path}')


def create_scenario_recipe_yaml(scenario_recipe, yaml_output_file_path):
    """
    Create YAML formatted scenario recipe file from which the scenario grid csv file can be created.

    Parameters
    ----------
    scenario_recipe : dict of array like
    yaml_output_file_path : str or Path, directory to which YAML scenario recipe file is written

    Returns
    -------
    No return value
    """

    # Create scenario lists from the grid specs above
    scenario_grid_lists = {key: value.tolist() for key, value in scenario_recipe.items()}

    with open(yaml_output_file_path, 'w') as f_yaml:
        yaml.dump(scenario_grid_lists, f_yaml, sort_keys=False)
        print(f'Scenario recipe YAML file written to {yaml_output_file_path}')


def process_command_line(argv=None):
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='scenario_tools',
                                     description='Create scenario related files for obflowsim')

    # Add arguments
    parser.add_argument(
        "exp", type=str,
        help="Experiment identifier. Used in filenames."
    )

    parser.add_argument(
        'scenario_recipe_yaml_file', type=str,
        help="Scenario recipe YAML file"
    )

    parser.add_argument(
        '-i', '--inputs_csv_path', type=str, default='./',
        help="Destination folder for simulation scenario input grid csv file"
    )

    # do the parsing
    args = parser.parse_args(argv)

    return args


def main(argv=None):
    # Parse command line arguments
    args = process_command_line(argv)

    csv_output_file_path = Path(args.inputs_csv_path, f'{args.exp}_obflowsim_scenario_inputs.csv')
    scenario_grid_to_csv(args.scenario_recipe_yaml_file, csv_output_file_path)


if __name__ == '__main__':
    sys.exit(main())
