# Create scenario inputs csv file
scenario_tools exp16b input/exp16b/exp16b_scenario_recipe.yaml -i input/exp16b/

# Copy the scenario input related files to the mm inputs
cp input/exp16b/exp16b_obflowsim_scenario_inputs.csv mm_input/exp16b
cp input/exp16b/exp16b_scenario_recipe.yaml mm_input/exp16b

# Generate predictions for exp16b using exp13 metamodels
mm_performance_curves exp13b exp16b \
    mm_input/exp16b/exp16b_obflowsim_scenario_inputs.csv \
    mm_output/exp13b \
    mm_input/exp13b \
    mm_output/exp16b


# Create simulation configuration files and run simulation script
create_configs exp16b \
    input/exp16b/exp16b_obflowsim_scenario_inputs.csv \
    input/exp16b/exp16b_obflowsim_settings.yaml \
    input/exp16b/config . --update_rho_checks

# Run simulations (this can take quite a while)      
sh ./exp16b_run.sh

# Concatenate the scenario replication files
obflow_io output/exp16b/stats/ output/exp16b/ summary_stats_scenario exp16b_scenario_rep_simout

# Create aggregate simulation summaries
obflow_stat output/exp16b/exp16b_scenario_rep_simout.csv output/exp16b exp16b --include_inputs --scenario_inputs_path input/exp16b/exp16b_obflowsim_scenario_inputs.csv

# Generate X, y matrices for metamodeling
mm_dataprep exp16b output/exp16b/scenario_siminout_exp16b.csv mm_input/exp16b/

# Merge predictions and simulation output (y vectors from mm_dataprep.py
mm_merge_predict_simulated exp16b \
    mm_output/exp16b/pc_predictions_exp16b_long.csv \
    mm_input/exp16b \
    mm_output/exp16b/predictions_simulated_exp16b_long.csv
