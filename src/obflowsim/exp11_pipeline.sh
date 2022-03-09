# Generate simulation inputs csv from scenario recipe YAML file
$CONDA_PREFIX/bin/python scenario_tools.py exp11 input/exp11/exp11_scenario_recipe.yaml -i input/exp11/

# Create config files for each scenario
$CONDA_PREFIX/bin/python create_configs.py exp11 \
        input/exp11/exp11_obflowsim_scenario_inputs.csv \
        input/exp11/exp11_obflowsim_settings.yaml \
        input/exp11/config run/exp11 500
        
# Run the simulation scenarios
sh ./run/exp11/exp11_run.sh

# Concatentate the scenario rep files - need to get these from config
$CONDA_PREFIX/bin/python obflow_io.py output/exp11/stats output/exp11/ summary_stats_scenario exp11_scenario_rep_simout

# Create the summary stats (aggregate over reps) by scenario
$CONDA_PREFIX/bin/python obflow_stat.py output/exp11/exp11_scenario_rep_simout.csv output/exp11 exp11 --include_inputs --scenario_inputs_path input/exp11/exp11_obflowsim_scenario_inputs.csv



