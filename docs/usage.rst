=====
Usage
=====

Setting up and running a multiscenario simulation experiment
-------------------------------------------------------------

The main steps are:

* Create the scenario input file
* Generate simulation config file for each scenario 
* Generate shell scripts to run the simulation scenarios
* Run the shell scripts to run the simulation scenarios
* Run obflow_io.py to concatenate the scenario rep files into one big scenario rep file
* Run obflow_stat.py to create the simulation summary files 

The input output summary file created after this final step
is ready to use in metamodel fitting and evaluation. The summary file
contains simulation inputs, outputs, and queueing approximations for
each scenario run.


Create the scenario input file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The scenario input file is generated from a *scenario recipe* - a YAML
formatted file that specifies one or more values for each of the simulation input
parameters. Here is an example:

.. code::
    arrival_rate:
    - 0.2
    - 0.6
    - 1.0
    mean_los_obs:
    - 2.0
    mean_los_ldr:
    - 12.0
    mean_los_csect:
    - 2.0
    mean_los_pp_noc:
    - 48.0
    mean_los_pp_c:
    - 72.0
    c_sect_prob:
    - 0.25
    num_erlang_stages_obs:
    - 1
    num_erlang_stages_ldr:
    - 2
    num_erlang_stages_csect:
    - 1
    num_erlang_stages_pp:
    - 8
    acc_obs:
    - 0.95
    - 0.99
    acc_ldr:
    - 0.85
    - 0.95
    acc_pp:
    - 0.85
    - 0.95

A few important things to note:

* The recipe file can be created manually or via the code in `scenario_scratchpad.py`. 
* The `acc_obs`, `acc_ldr`, and `acc_pp` accommodation probabilities lead to capacity lower bounds
based on an inverse Poisson approach. You can also directly specify `cap_obs`, `cap_ldr`,
and `cap_pp` capacity levels.

Assume you've create a scenario recipe file named `exp14_scenario_recipe.yaml`. Calling

.. code::
    python scenario_tools.py exp14 input/exp14/exp14_scenario_recipe.yaml -i input/exp14/
    
will generate the simuation scenario input file named `exp14_obflowsim_scenario_inputs.csv` in 
the `.inputs/exp14/` directory. Now we are ready to generate the configuration files for
each simulation scenario.
    
Generate simulation config file for each scenario
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``create_configs.py`` module does two main things:

* creates a config file for each simulation scenario
* generates shell scripts for running the simulation scenarios

.. code::

    create_configs [-h] [--update_rho_checks]
                          exp scenario_inputs_file_path sim_settings_file_path
                          configs_path run_script_path run_script_chunk_size

For example,

.. code::

    python create_configs.py exp14 \
        input/exp14/exp14_obflowsim_scenario_inputs.csv \
        input/exp14/exp14_obflowsim_settings.yaml \
        input/exp14/config run/exp14 500
                      
Generate shell scripts to run the simulation scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned in the previous step, ``create_configs.py`` creates the
shell scripts containing the commands to run the simulation scenarios. 
In order to take advantage of multiple CPUs, we can specify a 
``run_script_chunk_size`` parameter to break up the runs into multiple
scripts - each of which can be launched separately. It's a crude form
of parallel processing.

Run the shell scripts to run the simulation scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    sh ./run/exp14/exp14_run.sh

 
Run obflow_io.py to concatenate the scenario rep files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This will create the main output summary file with one row per (scenario, rep) pair.

.. code::

    obflow_io stop_summaries_path output_path summary_stats_file_stem \
                     output_file_stem

    
.. code::

    python obflow_io.py output/exp14/summary_stats/ output/exp14/ summary_stats_scenario exp14_scenario_rep_simout


Run obflow_stat.py to create the simulation summary files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At this point we have statistics for each (scenario, rep) pair and need to aggregate
over the replications to get stats by scenario.

.. code::
    obflow_stat [-h] [--process_logs] [--stop_log_path STOP_LOG_PATH]
                   [--occ_stats_path OCC_STATS_PATH] [--run_time RUN_TIME]
                   [--warmup_time WARMUP_TIME] [--include_inputs]
                   [--scenario_inputs_path SCENARIO_INPUTS_PATH]
                   scenario_rep_simout_path output_path suffix

.. code::

    python obflow_stat.py output/exp14/exp14_scenario_rep_simout.csv output/exp14 exp14 --include_inputs --scenario_inputs_path input/exp14/exp14_obflowsim_scenario_inputs.csv

Aggregates by scenario (over the replications).
Merges scenario inputs (which include the queueing approximations) with scenario simulation summary stats.

The input output summary file is ready to use in metamodeling experiments

Fitting and evaluation of simulation metamodels
-------------------------------------------------------------

The main steps in fitting metamodels are:

* Generate the X and y matrix data files from the simulation input output summary file. (mm_dataprep.py)
* Run the metamodel fits for OBS, LDR and PP (mm_run_fits_{unit}.py)
    - still need to add a CLI to these (added on 2022-02-25)
    - output includes metrics summary csv, actual vs predicted plots and coefficient plots
* Further output analysis (ongoing work)

Generating and evaluation of performance curves
-----------------------------------------------

Now that we have some good performing metamodels, we can use them to do things like generate
performance curves. The main steps are:

* Generation and evaluation of performance curves (ongoing work)
