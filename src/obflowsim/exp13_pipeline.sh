# Generate simulation inputs csv from scenario recipe YAML file
$CONDA_PREFIX/bin/python scenario_tools.py exp13 input/exp13/exp13_scenario_recipe.yaml -i input/exp13/

# Create config files for each scenario
$CONDA_PREFIX/bin/python create_configs.py exp13 \
        input/exp13/exp13_obflowsim_scenario_inputs.csv \
        input/exp13/exp13_obflowsim_settings.yaml \
        input/exp13/config run/exp13 500
        
# Run the simulation scenarios
sh ./run/exp13/exp13_run.sh

# Concatentate the scenario rep files - need to get these from config
$CONDA_PREFIX/bin/python obflow_io.py output/exp13/stats output/exp13/ summary_stats_scenario exp13_scenario_rep_simout

# Create the summary stats (aggregate over reps) by scenario
$CONDA_PREFIX/bin/python obflow_stat.py output/exp13/exp13_scenario_rep_simout.csv output/exp13 exp13 --include_inputs --scenario_inputs_path input/exp13/exp13_obflowsim_scenario_inputs.csv



