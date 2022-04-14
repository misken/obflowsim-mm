import sys
import argparse
from pathlib import Path


import pandas as pd
import yaml



def create_configs_from_inputs_csv(exp, scenarios_csv_file_path, simulation_settings_path, config_path,
                                   run_script_path, update_check_rho=False):
    """
    Create one simulation configuration file per scenario.
    
    Parameters
    ----------
    exp : str, experiment identifier
    scenarios_csv_file_path : str or Path, simulation scenario input csv file
    simulation_settings_path : str or Path, YAML file with simulation settings
    config_path : str or Path, destination for scenario specific config files
    run_script_path : str or Path, destination for shell scripts for running simulation scenarios
    update_check_rho : bool (Default=False), if True, recompute rho check values. Set to True if manual capacity levels set.

    Returns
    -------
    No return value
    """

    # Read scenarios file in DataFrame
    scenarios_df = pd.read_csv(scenarios_csv_file_path)
    # Read settings file
    with open(simulation_settings_path, 'rt') as settings_file:
        settings = yaml.safe_load(settings_file)
        #print(settings)

    global_vars = {}
    run_script_file_path = Path(run_script_path, f'{exp}_run.sh')
    with open(run_script_file_path, 'w') as bat_file:
        # Iterate over rows in scenarios file
        for row in scenarios_df.iterrows():
            scenario = int(row[1]['scenario'].tolist())

            global_vars['arrival_rate'] = row[1]['arrival_rate'].tolist()

            global_vars['mean_los_obs'] = row[1]['mean_los_obs'].tolist()
            global_vars['num_erlang_stages_obs'] = int(row[1]['num_erlang_stages_obs'])

            global_vars['mean_los_ldr'] = float(row[1]['mean_los_ldr'])
            global_vars['num_erlang_stages_ldr'] = int(row[1]['num_erlang_stages_ldr'])

            global_vars['mean_los_pp_noc'] = float(row[1]['mean_los_pp_noc'])
            global_vars['mean_los_pp_c'] = float(row[1]['mean_los_pp_c'])
            global_vars['num_erlang_stages_pp'] = int(row[1]['num_erlang_stages_pp'])

            global_vars['mean_los_csect'] = float(row[1]['mean_los_csect'])
            global_vars['num_erlang_stages_csect'] = int(row[1]['num_erlang_stages_csect'])

            global_vars['c_sect_prob'] = float(row[1]['c_sect_prob'])

            config = {}
            config['locations'] = settings['locations']
            cap_obs = int(row[1]['cap_obs'].tolist())
            cap_ldr = int(row[1]['cap_ldr'].tolist())
            cap_pp = int(row[1]['cap_pp'].tolist())
            config['locations'][1]['capacity'] = cap_obs
            config['locations'][2]['capacity'] = cap_ldr
            config['locations'][4]['capacity'] = cap_pp

            # Write scenario config file

            config['scenario'] = scenario
            config['run_settings'] = settings['run_settings']
            config['output'] = settings['output']
            config['random_number_streams'] = settings['random_number_streams']

            config['routes'] = settings['routes']
            config['global_vars'] = global_vars

            config_file_path = Path(config_path) / f'{exp}_scenario_{scenario}.yaml'

            with open(config_file_path, 'w', encoding='utf-8') as config_file:
                yaml.dump(config, config_file)

            run_line = f"obflow_sim {config_file_path} --loglevel=WARNING\n"
            bat_file.write(run_line)

        # Create output file processing line
        # output_proc_line = f'python obflow_stat.py {output_path_} {exp_suffix_} '
        # output_proc_line += f"--run_time {settings['run_settings']['run_time']} "
        # output_proc_line += f"--warmup_time {settings['run_settings']['warmup_time']} --include_inputs "
        # output_proc_line += f"--scenario_inputs_path {scenarios_csv_path_} --process_logs "
        # output_proc_line += f"--stop_log_path {settings['paths']['stop_logs']} "
        # output_proc_line += f"--occ_stats_path {settings['paths']['occ_stats']}"
        # bat_file.write(output_proc_line)

        # Update load and rho check values in case capacity levels were changed manually
        if update_check_rho:
            scenarios_df['check_load_obs'] = \
                scenarios_df.apply(lambda x: x.arrival_rate * x.mean_los_obs, axis=1)
            scenarios_df['check_load_ldr'] = \
                scenarios_df.apply(lambda x: x.arrival_rate * x.mean_los_ldr, axis=1)
            scenarios_df['check_load_pp'] = \
                scenarios_df.apply(
                    lambda x: x.arrival_rate * (
                                x.c_sect_prob * x.mean_los_pp_c + (1 - x.c_sect_prob) * x.mean_los_pp_noc),
                    axis=1)

            scenarios_df['check_rho_obs'] = \
                scenarios_df.apply(lambda x: round(x.check_load_obs / x.cap_obs, 2), axis=1)
            scenarios_df['check_rho_ldr'] = \
                scenarios_df.apply(lambda x: round(x.check_load_ldr / x.cap_ldr, 2), axis=1)
            scenarios_df['check_rho_pp'] = \
                scenarios_df.apply(lambda x: round(x.check_load_pp / x.cap_pp, 2), axis=1)

            # Rewrite scenarios input file with updated rho_checks
            scenarios_df.to_csv(scenarios_csv_file_path, index=False)

    print(f'Config files written to {Path(config_path)}')
    return run_script_file_path


def create_run_script_chunks(run_script_file_path, run_script_chunk_size):
    """
    Split shell script of simulation run commands into multiple files each
    (except for perhaps the last one) haveing ``bat_scenario_chunk_size`` lines.

    Parameters
    ----------
    run_script_file_path : str or Path
    run_script_chunk_size : int

    Returns
    -------
    No return value - creates multiple output files of simulation run commands.
    """

    base_script_path = Path(run_script_file_path).parent
    stem = Path(run_script_file_path).stem

    with open(run_script_file_path, 'r') as batf:
        bat_lines = batf.readlines()

    num_lines = len(bat_lines)
    num_full_chunks = num_lines // run_script_chunk_size

    if num_full_chunks == 0:
        start = 0
        end = num_lines
        chunk = bat_lines[slice(start, end)]
        chunk_bat_file = Path(base_script_path, f'{stem}_{start + 1}_{end}.sh')
        with open(chunk_bat_file, 'w') as chunkf:
            for line in chunk:
                chunkf.write(f'{line}')
    else:
        for i in range(num_full_chunks):
            start = i * run_script_chunk_size
            end = start + run_script_chunk_size
            chunk = bat_lines[slice(start, end)]

            chunk_bat_file = Path(base_script_path, f'{stem}_{start + 1}_{end}.sh')
            with open(chunk_bat_file, 'w') as chunkf:
                for line in chunk:
                    chunkf.write(f'{line}')

        # Write out any remaining partial chunks

        if end < num_lines - 1:
            start = end
            end = num_lines
            chunk_bat_file = Path(base_script_path, f'{stem}_{start + 1}_{end}.sh')
            chunk = bat_lines[start:]
            with open(chunk_bat_file, 'w') as chunkf:
                for line in chunk:
                    chunkf.write(f'{line}')


def process_command_line(argv=None):
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='create_configs',
                                     description='Create scenario related files for obflowsim')

    # Add arguments
    parser.add_argument(
        "exp", type=str,
        help="Experiment identifier. Used in filenames."
    )

    parser.add_argument(
        'scenario_inputs_file_path', type=str,
        help="Scenario inputs csv file"
    )

    parser.add_argument(
        'sim_settings_file_path', type=str,
        help="Simulation experiment settings YAML file"
    )

    parser.add_argument(
        'configs_path', type=str,
        help="Destination directory for the scenario config files"
    )

    parser.add_argument(
        'run_script_path', type=str,
        help="Destination directory for the scripts for running the simulations."
    )

    parser.add_argument(
        '--chunk_size', '-c', type=int, default=None,
        help="Number of run simulation commands in each script file."
    )

    parser.add_argument('--update_rho_checks', '-u', dest='update_rho', action='store_true',
                        help='Use flag if capacity levels explicitly set')

    # do the parsing
    args = parser.parse_args(argv)

    return args


def main(argv=None):
    # Parse command line arguments
    args = process_command_line(argv)

    run_script_file_path = create_configs_from_inputs_csv(args.exp, args.scenario_inputs_file_path,
                                                          args.sim_settings_file_path,
                                                          args.configs_path,
                                                          args.run_script_path, args.update_rho)

    if args.chunk_size:
        create_run_script_chunks(run_script_file_path, args.chunk_size)


if __name__ == '__main__':
    sys.exit(main())
