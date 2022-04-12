from setuptools import find_packages, setup

setup(
    name='obflowsim',
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        'pandas',
        'numpy',
        'pyyaml',
        'networkx',
        'statsmodels',
        'matplotlib',
        'seaborn',
        'sklearn'
    ],
    entry_points={  # Optional
        'console_scripts': [
            'scenario_tools=obflowsim.scenario_tools:main',
            'create_configs=obflowsim.create_configs:main',
            'obflow_sim=obflowsim.obflow_sim:main',
            'obflow_io=obflowsim.obflow_io:main',
            'obflow_stat=obflowsim.obflow_stat:main',
            'mm_dataprep=obflowsim.mm.mm_dataprep:main',
            'mm_run_fits_obs=obflowsim.mm.mm_run_fits_obs:main',
            'mm_run_fits_ldr=obflowsim.mm.mm_run_fits_ldr:main',
            'mm_run_fits_pp=obflowsim.mm.mm_run_fits_pp:main',
            'mm_process_fitted_models=obflowsim.mm.mm_process_fitted_models:main',
            'mm_performance_curves=obflowsim.mm.mm_performance_curves:main',
            'mm_merge_predict_simulated=obflowsim.mm.mm_merge_predict_simulated:main',
        ],
    },
    version='0.1.0',
    description='OB patient flow simulation in Python',
    author='misken',
    license='MIT',
)
