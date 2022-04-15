echo "Start: `date`"

# Create scenario inputs csv file
scenario_tools exp13b input/exp13b/exp13b_scenario_recipe.yaml -i input/exp13b/

# Create simulation configuration files and run simulation script
create_configs exp13b \
    input/exp13b/exp13b_obflowsim_scenario_inputs.csv \
    input/exp13b/exp13b_obflowsim_settings.yaml \
    input/exp13b/config . --update_rho_checks

# Run simulations (this can take quite a while)    
sh ./exp13b_run_1_500.sh &
sh ./exp13b_run_501_1000.sh &
sh ./exp13b_run_1001_1500.sh &
wait
sh ./exp13b_run_1501_2000.sh &
sh ./exp13b_run_2001_2500.sh &
sh ./exp13b_run_2501_3000.sh &
wait
sh ./exp13b_run_3001_3240.sh


# Concatenate the scenario replication files
obflow_io output/exp13b/stats/ output/exp13b/ summary_stats_scenario exp13b_scenario_rep_simout

# Create aggregate simulation summaries
obflow_stat output/exp13b/exp13b_scenario_rep_simout.csv output/exp13b exp13b --include_inputs --scenario_inputs_path input/exp13b/exp13b_obflowsim_scenario_inputs.csv

# Generate X, y matrices for metamodeling
mm_dataprep exp13b output/exp13b/scenario_siminout_exp13b.csv mm_input/exp13b/

# Do the metamodel fits in parallel processes
mm_run_fits_obs exp13b mm_input/exp13b/ mm_output/exp13b/ mm_output/exp13b/plots/ &
mm_run_fits_ldr exp13b mm_input/exp13b/ mm_output/exp13b/ mm_output/exp13b/plots/ &
mm_run_fits_pp exp13b mm_input/exp13b/ mm_output/exp13b/ mm_output/exp13b/plots/ &
wait

echo "End: `date`"
