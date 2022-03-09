# Generate simulation inputs csv from scenario recipe YAML file
$CONDA_PREFIX/bin/python scenario_tools.py exp15 input/exp15/exp15_scenario_recipe.yaml -i input/exp15/

# Create config files for each scenario
$CONDA_PREFIX/bin/python create_configs.py exp15 \
        input/exp15/exp15_obflowsim_scenario_inputs.csv \
        input/exp15/exp15_obflowsim_settings.yaml \
        input/exp15/config run/exp15 500
        
# Run the simulation scenarios
sh ./run/exp15/exp15_run.sh

# Concatentate the scenario rep files - need to get these from config
$CONDA_PREFIX/bin/python obflow_io.py output/exp15/stats output/exp15/ summary_stats_scenario exp15_scenario_rep_simout

# Create the summary stats (aggregate over reps) by scenario
$CONDA_PREFIX/bin/python obflow_stat.py output/exp15/exp15_scenario_rep_simout.csv output/exp15 exp15 --include_inputs --scenario_inputs_path input/exp15/exp15_obflowsim_scenario_inputs.csv



