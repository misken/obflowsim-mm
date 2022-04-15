# obflowsim - a research project


The obflowsim Python package is a discrete event simulation model built with
[SimPy](https://simpy.readthedocs.io/en/latest/) of a very simple inpatient obstetrical patient flow network. In
addition to the DES model there are modules for related simulation
input and output data processing including fitting of metamodels. 

The 
obflowsim package was developed as part of a research project involving
comparison of simulation metamodeling methods and the impact of feature
engineering on metamodel accuracy and explainability. We are making this repository public
so that anyone can reproduce our results and to provide full
transparency as to how the simulation model works and the metamodels
were fit.

The following blog posts I have done over the past few years provide some background on simulation with SimPy
and building simulation metamodels for this patient flow network.

- [Getting started with SimPy for patient flow modeling](https://misken.github.io/blog/simpy-getting-started/)
- [An object oriented SimPy patient flow simulation model](https://misken.github.io/blog/simpy-first-oo-patflow-model/)
- [Comparing predictive models for obstetrical unit occupancy using caret - Part 1](https://misken.github.io/blog/obsim_caret_part1/)
- [Comparing predictive model performance using caret - Part 2: A simple caret automation function](https://misken.github.io/blog/obsim_caret_part2/)
- [Comparing predictive model performance using caret - Part 3: Put it all together](https://misken.github.io/blog/obsim_caret_part3/)


If you want to explore the code in this project and/or rerun any
of the analysis that led to the results presented in our paper, there
are a few steps. 

## Getting and installing obflowsim

Clone the obflowsim project.

    $ git clone git@github.com:misken/obflowsim.git
    
Create a conda virtual env using the `obflowsim.yml` file.

    $ conda env create -f obflowsim.yml
    $ conda activate obflowsim
    
Pip install `obflowsim` using the `pip` executable installed in the 
`obflowsim` conda virtual environment.

    $ $CONDA_PREFIX/bin/pip install -e .
    
## Explore the explainer notebook

There's a Jupyter notebook named `obflowsim_explainer.ipynb` in the
`notebooks` folder. It walks through all the steps of running a simulation
experiment and fitting metamodels from the simulation output.

## Reproduce the analysis done for the paper

In the `run_paper_experiments` folder you'll find (among other things)
three "pipeline" bash shell scripts:

- `exp11b_pipeline.sh` - runs the experiment referred to as EXP1 in our paper,
- `exp13b_pipeline.sh` - runs the experiment referred to as EXP2 in our paper,
- `exp16b_pipeline.sh` - runs the experiment referred to as EXP3 in our paper.

**NOTE** These experiments take quite a while to run. See the comments
in the pipeline scripts. For example, `exp13b_pipeline.sh` involves
over 3000 simulation scenarios (each replicated 25 times) and the
total run time is probably 10-20 hours depending on the number of CPUs used.

The scripts are commented and all of the folder structure needed for running
the scripts has already been set up. There are also some R Markdown documents
a R scripts used to produce some of the figures in the paper.

        ├── exp11b_paper_plots.Rmd
        ├── exp11b_pipeline.sh
        ├── exp11b_run.sh
        ├── exp13bm11b_diff_plots.Rmd
        ├── exp13b_paper_plots.Rmd
        ├── exp13b_pipeline.sh
        ├── exp13b_run_1001_1500.sh
        ├── exp13b_run_1_500.sh
        ├── exp13b_run_1501_2000.sh
        ├── exp13b_run_2001_2500.sh
        ├── exp13b_run_2501_3000.sh
        ├── exp13b_run_3001_3240.sh
        ├── exp13b_run_501_1000.sh
        ├── exp16b_pipeline.sh
        ├── input
        │   ├── exp11b
        │   │   ├── config
        │   │   │   └── README.md
        │   │   ├── exp11b_obflowsim_settings.yaml
        │   │   └── exp11b_scenario_recipe.yaml
        │   ├── exp13b
        │   │   ├── config
        │   │   │   └── README.md
        │   │   ├── exp13b_obflowsim_settings.yaml
        │   │   └── exp13b_scenario_recipe.yaml
        │   └── exp16b
        │       ├── config
        │       │   └── README.md
        │       ├── exp16b_obflowsim_settings.yaml
        │       └── exp16b_scenario_recipe.yaml
        ├── make_diff_plots.R
        ├── make_plots.R
        ├── mm_input
        │   ├── exp11b
        │   │   └── README.md
        │   ├── exp13b
        │   │   └── README.md
        │   └── exp16b
        │       └── README.md
        ├── mm_output
        │   ├── exp11b
        │   │   └── plots
        │   │       └── README.md
        │   ├── exp13b
        │   │   └── plots
        │   │       └── README.md
        │   └── exp16b
        │       └── plots
        │           └── README.md
        ├── new_exp.sh
        └── output





