echo "Start: `date`"

# Create scenario inputs csv file
scenario_tools exp11b input/exp11b/exp11b_scenario_recipe.yaml -i input/exp11b/

# Create simulation configuration files and run simulation script
create_configs exp11b \
    input/exp11b/exp11b_obflowsim_scenario_inputs.csv \
    input/exp11b/exp11b_obflowsim_settings.yaml \
    input/exp11b/config . --update_rho_checks

# Run simulations (this can take quite a while)    
sh ./exp11b_run.sh

# Concatenate the scenario replication files
obflow_io output/exp11b/stats/ output/exp11b/ summary_stats_scenario exp11b_scenario_rep_simout

# Create aggregate simulation summaries
obflow_stat output/exp11b/exp11b_scenario_rep_simout.csv output/exp11b exp11b --include_inputs --scenario_inputs_path input/exp11b/exp11b_obflowsim_scenario_inputs.csv

# Generate X, y matrices for metamodeling
mm_dataprep exp11b output/exp11b/scenario_siminout_exp11b.csv mm_input/exp11b/

# Do the metamodel fits in parallel processes
mm_run_fits_obs exp11b mm_input/exp11b/ mm_output/exp11b/ mm_output/exp11b/plots/ &
mm_run_fits_ldr exp11b mm_input/exp11b/ mm_output/exp11b/ mm_output/exp11b/plots/ &
mm_run_fits_pp exp11b mm_input/exp11b/ mm_output/exp11b/ mm_output/exp11b/plots/ &
wait

echo "End: `date`"
