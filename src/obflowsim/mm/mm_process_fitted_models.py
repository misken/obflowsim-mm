from pathlib import Path
import pickle
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def create_cv_plots(experiment, unit, results_dict, figures_path):

    for key in results_dict.keys():
        print(f"cv plot: {key}")
        scenario = results_dict[key]['scenario']
        scatter_plot = results_dict[key]['fitplot']
        plot_name = f"{experiment}_{unit}_{scenario}_cv_scatter.png"
        scatter_plot.savefig(Path(figures_path, plot_name))


def create_coeff_plots(experiment, unit, results_dict, figures_path):
    for key in results_dict.keys():
        scenario = results_dict[key]['scenario']
        if 'coefplot' in results_dict[key].keys():
            print(f"coeff plot: {key}")
            scatter_plot = results_dict[key]['coefplot']
            plot_name = f"{experiment}_{unit}_{scenario}_cv_coeff.png"
            scatter_plot.savefig(Path(figures_path, plot_name))


def create_metrics_df(results_dict, output_path):
    dfs = []
    for key in results_dict.keys():
        print(f"metrics df: {key}")
        scenario = results_dict[key]['scenario']

        metrics_df = results_dict[key]['metrics_df']
        metrics_df['scenario'] = scenario
        metrics_df['flavor'] = results_dict[key]['flavor']
        metrics_df['unit'] = results_dict[key]['unit']
        dfs.append(metrics_df)

    consolidated_metrics_df = pd.concat(dfs)
    consolidated_metrics_df.reset_index(inplace=True)
    consolidated_metrics_df.rename(columns={'index': 'fold'}, inplace=True)
    return consolidated_metrics_df

def process_command_line():
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parsercorrect
    parser = argparse.ArgumentParser(prog='process_fitted_models',
                                     description='Create plots and summaries for fitted models')

    # Add arguments
    parser.add_argument(
        "experiment", type=str,
        help="String used in output filenames"
    )

    parser.add_argument(
        "unit", type=str,
        help="String used in summary dataframes"
    )

    parser.add_argument(
        "pkl_to_process", type=str,
        help="Pickle filename containing model fit results"
    )

    parser.add_argument(
        "output_path", type=str,
        help="Directory to write summaries"
    )

    parser.add_argument(
        "figures_path", type=str,
        help="Directory to write figures"
    )

    # do the parsing
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    override_args = True

    if override_args:
        experiment = "exp11"
        unit = "ldr"
        pkl_to_process = f"{unit}_results_exp11.pkl"
        output_path = "output"
        figures_path = f"output/figures/{experiment}"
    else:
        pfm_args = process_command_line()
        experiment = pfm_args.experiment
        unit = pfm_args.unit
        pkl_to_process = pfm_args.pkl_to_process
        output_path = pfm_args.output_path
        figures_path = pfm_args.figures_path

    with open(Path(output_path, pkl_to_process), 'rb') as pickle_file:
        pickeled_results = pickle.load(pickle_file)
        create_cv_plots(experiment, unit, pickeled_results, Path(figures_path))
        create_coeff_plots(experiment, unit, pickeled_results, Path(figures_path))

        metrics_df = create_metrics_df(pickeled_results, Path("output_path"))
        metrics_df.to_csv(Path(output_path, f"{experiment}_{unit}_metrics_df.csv"), index=False)



