from pathlib import Path
import pickle
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def create_cv_plots(experiment, unit, results_dict, figures_path):

    for key in results_dict.keys():
        print(f"cv plot: {key}")
        unit_pm_qdata_model = results_dict[key]['unit_pm_qdata_model']
        scatter_plot = results_dict[key]['fitplot']
        plot_name = f"{experiment}_{unit}_{unit_pm_qdata_model}_cv_scatter.png"
        scatter_plot.savefig(Path(figures_path, plot_name))


def create_coeff_plots(experiment, unit, results_dict, figures_path):
    for key in results_dict.keys():
        unit_pm_qdata_model = results_dict[key]['unit_pm_qdata_model']
        if 'coefplot' in results_dict[key].keys():
            print(f"coeff plot: {key}")
            scatter_plot = results_dict[key]['coefplot']
            plot_name = f"{experiment}_{unit}_{unit_pm_qdata_model}_cv_coeff.png"
            scatter_plot.savefig(Path(figures_path, plot_name))


def create_metrics_df(results_dict):
    dfs = []
    for key in results_dict.keys():
        print(f"metrics df: {key}")
        unit_pm_qdata_model = results_dict[key]['unit_pm_qdata_model']

        metrics_df = results_dict[key]['metrics_df']
        metrics_df['unit_pm_qdata_model'] = unit_pm_qdata_model
        metrics_df['unit'] = results_dict[key]['unit']
        metrics_df['measure'] = results_dict[key]['measure']
        metrics_df['qdata'] = results_dict[key]['qdata']
        metrics_df['model'] = results_dict[key]['flavor']

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
        experiment = "exp13"
        unit = "pp"
        output_path = f"output/{experiment}"
        pickle_path_filename = Path(output_path, f"{experiment}_{unit}_results.pkl")

        figures_path = f"output/{experiment}/figures"
    else:
        pfm_args = process_command_line()
        experiment = pfm_args.experiment
        unit = pfm_args.unit
        output_path = pfm_args.output_path
        pkl_to_process = pfm_args.pkl_to_process
        pickle_path_filename = Path(output_path, pkl_to_process)

        figures_path = pfm_args.figures_path

    with open(pickle_path_filename, 'rb') as pickle_file:
        results_dict = pickle.load(pickle_file)
        create_cv_plots(experiment, unit, results_dict, Path(figures_path))
        create_coeff_plots(experiment, unit, results_dict, Path(figures_path))

        metrics_df = create_metrics_df(results_dict)
        metrics_df.to_csv(Path(output_path, f"{experiment}_{unit}_metrics.csv"), index=False)



