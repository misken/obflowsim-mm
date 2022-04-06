from setuptools import find_packages, setup

setup(
    name='obflowsim',
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'pyyaml',
    ],
    entry_points={  # Optional
        'console_scripts': [
            'scenario_tools=obflowsim.scenario_tools:main',
            'create_configs=obflowsim.create_configs:main',
            'obflow_sim=obflowsim.obflow_sim:main',
            'obflow_io=obflowsim.obflow_io:main',
            'obflow_stat=obflowsim.obflow_stat:main',
        ],
    },
    version='0.1.0',
    description='OB patient flow simulation in Python',
    author='misken',
    license='MIT',
)
