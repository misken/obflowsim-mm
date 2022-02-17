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

The scenario input file is a csv file containing (at least) the following columns:

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
* The `acc_obs`, `acc_ldr`, and `acc_pp` accomodation probabilities lead to capacity lower bounds
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

The ``create_configs.py`` module needs a CLI.

.. code::

    python create_configs.py output/exp13/summary_stats/ output/exp13/ summary_stats_scenario exp13_scenario_rep_simout
    
Generate shell scripts to run the simulation scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the shell scripts to run the simulation scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 
Run obflow_io.py to concatenate the scenario rep files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
.. code::

    python obflow_io.py output/exp13/summary_stats/ output/exp13/ summary_stats_scenario exp13_scenario_rep_simout

Run obflow_stat.py to create the simulation summary files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    python obflow_stat.py output/exp13/exp13_scenario_rep_simout.csv output/exp13 exp13 --include_inputs --scenario_inputs_path input/exp13/exp13_obflowsim_metainputs.csv

Aggregates by scenario (over the replications).
Merges scenario inputs with scenario simulation summary stats.
Computes queueing approximations to include with with input output summary.

The input output summary file is ready to use in metamodeling experiments

Fitting and evaluation simulation metamodels
-------------------------------------------------------------
