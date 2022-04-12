=====
Usage
=====

.. highlight:: bash

Setting up and running a multiscenario simulation experiment
-------------------------------------------------------------

The main steps are:

* Create the scenario input file (``scenario_tools``)
* Create the run settings file (manually for now)
* Generate simulation config file and for each scenario and shell scripts to run the scenarios (``create_configs``)
* Run the shell scripts to run the simulation scenarios (``obflow_sim``)
* Concatenate the scenario rep files into one big scenario rep file(``obflow_io``)
* Create the simulation summary files (``obflow_stat``)

The input output summary file created after this final step
is ready to use in metamodel fitting and evaluation. The summary file
contains simulation inputs, outputs, and queueing approximations for
each scenario run. The main metamodeling steps are:

* Create the X and y matrices for metamodel fitting (``mm_dataprep``)
* Fit models (``mm_run_fits_obs``, ``mm_run_fits_ldr``, and ``mm_run_fits_pp``)
* Postprocess the fitted models (``mm_process_fitted_models``)
* Generate performance curves from fitted models (``mm_performance_curves``)


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

* The recipe file can be created manually or via the code in ``scenario_scratchpad.py``.
* The ``acc_obs``, ``acc_ldr``, and ``acc_pp`` accommodation probabilities lead to capacity lower bounds
based on an inverse Poisson approach. You can also directly specify `cap_obs`, `cap_ldr`,
and ``cap_pp`` capacity levels.

Assume you've create a scenario recipe file named ``exp14_scenario_recipe.yaml``. Calling

.. code::
    scenario_tools exp14 input/exp14/exp14_scenario_recipe.yaml -i input/exp14/
    
will generate the simulation scenario input file named ``exp14_obflowsim_scenario_inputs.csv`` in
the ``.inputs/exp14/`` directory. Now we are ready to generate the configuration files for
each simulation scenario.

Create run settings file and setup directory structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO

Generate simulation config file for each scenario
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``create_configs.py`` module does two main things:

* creates a config file for each simulation scenario
* generates shell scripts for running the simulation scenarios

.. code::

    usage: create_configs [-h] [--chunk_size CHUNK_SIZE] [--update_rho_checks]
                      exp scenario_inputs_file_path sim_settings_file_path
                      configs_path run_script_path

For example,

.. code::

    create_configs exp14 \
        input/exp14/exp14_obflowsim_scenario_inputs.csv \
        input/exp14/exp14_obflowsim_settings.yaml \
        input/exp14/config run/exp14 --chunk_size 500 --update_rho_checks

Set ``--update_rho_checks`` if you manually set capacity levels in the scenario inputs file. This
will help you detect scenarios with insufficient capacity (i.e. $\rho > 1$).
                      
Generate shell scripts to run the simulation scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned in the previous step, ``create_configs.py`` creates the
shell scripts containing the commands to run the simulation scenarios. 
In order to take advantage of multiple CPUs, we can specify a 
``--chunk_size`` parameter to break up the runs into multiple
scripts - each of which can be launched separately. It's a crude form
of parallel processing.

Run the shell scripts to run the simulation scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single scenario can be run by using ``obflow_sim``.

.. code::
    usage: obflow_io [-h] stop_summaries_path output_path summary_stats_file_stem output_file_stem

    Run inpatient OB simulation

    positional arguments:
      stop_summaries_path   Folder containing the scenario rep summaries created by simulation runs
      output_path           Destination folder for combined scenario rep summary csv
      summary_stats_file_stem
                            Summary stat file name without extension
      output_file_stem      Combined summary stat file name without extension to be output

    optional arguments:
      -h, --help            show this help message and exit
    (obflowsim) mark@quercus:~/Documents/research/OBsim/mm_interpet/rerun25$ obflow_sim -h
    usage: obflow_6 [-h] [--loglevel LOGLEVEL] config

    Run inpatient OB simulation

    positional arguments:
      config               Configuration file containing input parameter arguments and values

    optional arguments:
      -h, --help           show this help message and exit
      --loglevel LOGLEVEL  Use valid values for logging package



.. code::
    obflow_sim input/exp14/config/exp14_scenario_1.yaml

The shell scripts generated in the previous step are just a sequence of such
single scenario command lines.

.. code::

    sh ./run/exp14/exp14_run.sh

 
Run ``obflow_io`` to concatenate the scenario replication files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This will create the main output summary file with one row per (scenario, replication) pair.

.. code::

    usage: obflow_io [-h] stop_summaries_path output_path summary_stats_file_stem output_file_stem

    create the main output summary file with one row per (scenario, replication) pair

    positional arguments:
      stop_summaries_path   Folder containing the scenario rep summaries created by simulation runs
      output_path           Destination folder for combined scenario rep summary csv
      summary_stats_file_stem
                            Summary stat file name without extension
      output_file_stem      Combined summary stat file name without extension to be output

    optional arguments:
      -h, --help            show this help message and exit

    
.. code::

    obflow_io output/exp14/summary_stats/ output/exp14/ summary_stats_scenario exp14_scenario_rep_simout


Run ``obflow_stat`` to create the simulation summary files
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

    obflow_stat output/exp14/exp14_scenario_rep_simout.csv output/exp14 exp14 --include_inputs --scenario_inputs_path input/exp14/exp14_obflowsim_scenario_inputs.csv

Aggregates by scenario (over the replications).
Merges scenario inputs (which include the queueing approximations) with scenario simulation summary stats.

The input output summary file is ready to use in metamodeling experiments. It will
be named ``scenario_siminout_{experiment id}.csv``. Continuing our example, the output
file is ``scenario_siminout_exp14.csv``


Fitting and evaluation of simulation metamodels
-------------------------------------------------------------

The main steps in fitting metamodels are:

* Generate the X and y matrix data files from the simulation input output summary file. (``mm_dataprep``)
* Run the metamodel fits for OBS, LDR and PP (``mm_run_fits_{unit}``)
    - output includes metrics summary csv, actual vs predicted plots and coefficient plots
* Generate performance curves (``mm_performance_curves``)

Generate the X and y matrix data files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The specific columns (independent variables) in the ``X`` matrices and ``y`` target vectors are driven by the
project design and goals. See ``mm_dataprep.py``.

.. code::
    usage: mm_dataprep [-h] experiment siminout_qng_path output_data_path

    Create X and y files for metamodeling

    positional arguments:
      experiment         String used in output filenames
      siminout_qng_path  Path to csv file which will contain scenario inputs, summary stats and qng approximations
      output_data_path   Path to directory in which to create X and y data files

    optional arguments:
      -h, --help         show this help message and exit

.. code::

    mm_dataprep exp14 input/exp14_obflowsim_scenario_inputs.csv mmdata/exp14/

Metamodel fitting
^^^^^^^^^^^^^^^^^^^^^^

See ``mm_run_fits_obs.py``, ``mm_run_fits_ldr.py``, and ``mm_run_fits_pp.py`` for details. The core
model fitting procedure is part of ``mm_fitting.py``.

.. code::

    mm_run_fits_obs mmdata/exp14/ mmoutput/exp14/ mmoutput/exp14/plots/
    mm_run_fits_ldr mmdata/exp14/ mmoutput/exp14/ mmoutput/exp14/plots/
    mm_run_fits_pp mmdata/exp14/ mmoutput/exp14/ mmoutput/exp14/plots/

The output includes a pickle file containing detailed model fitting results.
Predicted vs. actual plots as well as coefficient plots are also created. Here's
what the pickled dictionary looks like (with comments). Notice that some
items might not be applicable for all model flavors. For example, random forest
models do not have coefficient values.

.. code-block:: python
    results = {'unit_pm_qdata_model': unit_pm_qdata_model, # Composite identifier
                       'measure': measure, # str: performance measure
                       'flavor': flavor,   # str: model flavor
                       'unit': unit,       # str: hospital unit
                       'var_names': var_names, # list: column names of predictors
                       'model': model_final,   # sklearn model: final fitted model
                       'qdata' : qdata,        # str: level of queueing inputs
                       'cv': cv_iterator,      # sklearn cv iterator: cv details
                       'coeffs_df': unscaled_coeffs_df, # dataframe: unscaled coeffs (if applicable)
                       'metrics_df': metrics_df,        # dataframe: metrics by fold
                       'scaling': scaling_factors,      # numpy array: scaling factors for X
                       'scaled_coeffs_df': coeffs_df,   # dataframe: scaled coeffs (if applicable)
                       'alphas': alphas,       # numpy array: penalization values for lassocv
                       'predictions': predictions, # numpy array: predicted values on leave out observations
                       'residuals': residuals,     # numpy array: predicted - actual on leave out observations
                       'fitplot': fig_scatter,     # matplotlib Figure: scatter pred vs act
                       'coefplot': fig_coeffs}     # matplotlib Figure: coefficients by fold (if applicable)

Generating and evaluation of performance curves
-----------------------------------------------

Now that we have some good performing metamodels, we can use them to do things like generate
performance curves.

.. code::
    usage: mm_performance_curves [-h]
                                 mm_experiment predict_experiment scenario_input_path_filename pkl_path X_data_path
                                 output_path

    Generate predictions from fitted models

    positional arguments:
      mm_experiment         Experiment used to fit metamodels
      predict_experiment    Experiment for which to predict
      scenario_input_path_filename
                            Path to csv file which contains scenario inputs
      pkl_path              Path containing pkl files created from metamodel fitting
      X_data_path           Path to directory in which to write X data for predictions
      output_path           Path to write output csv files

    optional arguments:
      -h, --help            show this help message and exit

In the example below, we are using models fitted in ``exp14`` to make predictions
for scenarios in ``exp15``.

.. code::

    # Generate scenario input file
    scenario_tools exp15 input/exp15/exp15_scenario_recipe.yaml -i input/exp15/

    # Generate performance curves
    mm_performance_curves exp14 exp15 \
        mm_input/exp15/exp15_obflowsim_scenario_inputs.csv \
        mm_output/exp14 \
        mm_input/exp14 \
        mm_output/exp15

If we want to assess the accuracy of these predictions, we just need to run the simulation
model for the scenarios in ``exp15``.

.. code::
    # Create config files and scripts for running simulations
    create_configs exp15 \
    input/exp15/exp15_obflowsim_scenario_inputs.csv \
    input/exp15/exp15_obflowsim_settings.yaml \
    input/exp15/config . --update_rho_checks
    # Run simulations
    sh ./exp15_run.sh
    # Combine scenario specific output files
    obflow_io output/exp15/stats/ output/exp15/ summary_stats_scenario exp15_scenario_rep_simout
    # Compute aggregated (over replications) output stats
    obflow_stat output/exp15/exp15_scenario_rep_simout.csv output/exp15 exp15 --include_inputs --scenario_inputs_path input/exp15/exp15_obflowsim_scenario_inputs.csv

To facilitate comparing of predicted vs actual values for the new scenarios, we can combine the
simulation output with the metamodel predictions using ``mm_merge_predict_simulated``.

.. code::
    usage: mm_merge_predict_simulated [-h] experiment perf_curve_pred_filename y_data_path output_filename

    merge simulation scenario output with mm predictions

    positional arguments:
      experiment            String used in output filenames
      perf_curve_pred_filename
                            Path to csv file which contains predictions
      y_data_path           Path to directory containing y data files (which are created from sim output)
      output_filename       Path to merged output csv file

    optional arguments:
      -h, --help            show this help message and exit



.. code::
    # Merge predictions and simulation output
    mm_merge_predict_simulated exp15 \
    mm_output/exp15/pc_predictions_exp15_long.csv \
    mm_input/exp15 \
    mm_output/exp15/predictions_simulated_exp15_long.csv