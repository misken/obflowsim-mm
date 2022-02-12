from pathlib import Path

import pandas as pd
import numpy as np
import yaml


def config_from_csv(exp_suffix_,
                    scenarios_csv_path_, settings_path_, config_path_,
                    output_path_, bat_path_, bat_chunk_size_=500):

    # Read scenarios file in DataFrame
    scenarios_df = pd.read_csv(scenarios_csv_path_)
    # Read settings file
    with open(settings_path_, 'rt') as settings_file:
        settings = yaml.safe_load(settings_file)
        print(settings)

    global_vars = {}
    with open(bat_path_, 'w') as bat_file:
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

            config_file_path = Path(config_path_) / f'scenario_{scenario}.yaml'

            with open(config_file_path, 'w', encoding='utf-8') as config_file:
                yaml.dump(config, config_file)

            run_line = f"python obflow_sim.py {config_file_path} --loglevel=WARNING\n"
            bat_file.write(run_line)

        # Create output file processing line
        # output_proc_line = f'python obflow_stat.py {output_path_} {exp_suffix_} '
        # output_proc_line += f"--run_time {settings['run_settings']['run_time']} "
        # output_proc_line += f"--warmup_time {settings['run_settings']['warmup_time']} --include_inputs "
        # output_proc_line += f"--scenario_inputs_path {scenarios_csv_path_} --process_logs "
        # output_proc_line += f"--stop_log_path {settings['paths']['stop_logs']} "
        # output_proc_line += f"--occ_stats_path {settings['paths']['occ_stats']}"
        # bat_file.write(output_proc_line)


def create_bat_chunks(bat_path_, bat_chunk_size_):

    base_bat_path = Path(bat_path_).parent
    stem = Path(bat_path_).stem

    with open(bat_path_, 'r') as batf:
        bat_lines = batf.readlines()

    num_lines = len(bat_lines)
    num_full_chunks = num_lines // bat_chunk_size_

    for i in range(num_full_chunks):
        start = i  * bat_chunk_size_
        end = start + bat_chunk_size_
        chunk = bat_lines[slice(start, end)]

        chunk_bat_file = Path(base_bat_path, f'{stem}_{start + 1}_{end + 1}.sh')
        with open(chunk_bat_file, 'w') as chunkf:
            for line in chunk:
                chunkf.write(f'{line}')

    # Write out any remaining partial chunks

    if end < num_lines - 1:
        start = end
        end = num_lines
        chunk_bat_file = Path(base_bat_path, f'{stem}_{start + 1}_{end + 1}.sh')
        chunk = bat_lines[start:]
        with open(chunk_bat_file, 'w') as chunkf:
            for line in chunk:
                chunkf.write(f'{line}')


if __name__ == '__main__':

    exp_suffix = 'exp13'
    scenarios_csv_path = Path(f'input/{exp_suffix}/{exp_suffix}_obflowsim_metainputs.csv')
    settings_path = Path(f'input/{exp_suffix}/{exp_suffix}_obflowsim_settings.yaml')
    config_path = Path(f'input/{exp_suffix}/config/')
    bat_path = Path('./run') / f'{exp_suffix}_obflowsim_run.sh'
    output_path = Path('./output') / f'{exp_suffix}/'


    config_from_csv(exp_suffix,
                    scenarios_csv_path, settings_path, config_path,
                    output_path, bat_path)

    bat_chunk_size = 500
    create_bat_chunks(bat_path, bat_chunk_size)
