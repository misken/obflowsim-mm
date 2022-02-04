from setuptools import find_packages, setup

setup(
    name='obflowsim',
    packages=find_packages("src"),
    package_dir={"": "src"},
    entry_points={  # Optional
        'console_scripts': [
            'obflowsim=obflowsim.obflowsim:runsim',
        ],
    },
    version='0.1.0',
    description='OB patient flow simulation in Python',
    author='misken',
    license='MIT',
)
