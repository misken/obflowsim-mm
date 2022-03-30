from pathlib import Path
import pickle
import argparse

import numpy as np
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


def output_model_coeffs(experiment, results_dict, output_path):
    """NOT IMPLEMENTED"""

    for key in results_dict.keys():
        unit_pm_qdata_model = results_dict[key]['unit_pm_qdata_model']

        if 'coeffs_df' in results_dict[key].keys():
            # model = results_dict[key]['model']
            # flavor = results_dict[key]['flavor']
            coeffs_df = results_dict[key]['coeffs_df']
            coeffs_df.to_csv(Path(output_path, f"{experiment}_{unit_pm_qdata_model}_coeffs.csv"), index=False)




def create_predictions_df(results_dict):
    dfs = []
    for key in results_dict.keys():
        print(f"results df: {key}")
        unit_pm_qdata_model = results_dict[key]['unit_pm_qdata_model']

        predictions_df = pd.DataFrame()
        num_scenarios = len(results_dict[key]['predictions'])
        predictions_df['scenario'] = np.arange(1, num_scenarios + 1)
        predictions_df['prediction'] = results_dict[key]['predictions']
        resids = results_dict[key]['residuals'].to_numpy()
        predictions_df['actual'] = results_dict[key]['predictions'] - resids
        predictions_df['unit_pm_qdata_model'] = unit_pm_qdata_model
        predictions_df['unit'] = results_dict[key]['unit']
        predictions_df['measure'] = results_dict[key]['measure']
        predictions_df['qdata'] = results_dict[key]['qdata']
        predictions_df['model'] = results_dict[key]['flavor']

        dfs.append(predictions_df)

    consolidated_predictions_df = pd.concat(dfs)
    consolidated_predictions_df.reset_index(inplace=True)
    return consolidated_predictions_df


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


def get_feature_importance(results_dict):

    scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error']

    for key in results_dict.keys():
        unit_pm_qdata_model = results_dict[key]['unit_pm_qdata_model']

        if 'importance' in results_dict[key].keys():
            print(unit_pm_qdata_model)
            importance = results_dict[key]['importance']
            feature_names = results_dict[key]['var_names']

            for metric in importance:
                print(f"{metric}")
                r = importance[metric]
                for i in r.importances_mean.argsort()[::-1]:
                    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                        print(f"    {feature_names[i]:<8}"
                              f"{r.importances_mean[i]:.3f}"
                              f" +/- {r.importances_std[i]:.3f}")


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
        help="Pickle path and filename containing model fit results"
    )

    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Directory to write summaries"
    )

    parser.add_argument(
        "--figures_path", type=str, default=None,
        help="Directory to write figures"
    )

    parser.add_argument(
        "--importance", action='store_true',
        help="Include to report importance values for rf and nnet"
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
        output_path = None
        pkl_to_process = Path(f"output/{experiment}", f"{experiment}_{unit}_results.pkl")
        #figures_path = f"output/{experiment}/figures"
        figures_path = None
        importance = True
    else:
        pfm_args = process_command_line()
        experiment = pfm_args.experiment
        unit = pfm_args.unit
        output_path = pfm_args.output_path
        importance = pfm_args.importance
        pkl_to_process = pfm_args.pkl_to_process
        pickle_path_filename = Path(output_path, pkl_to_process)
        figures_path = pfm_args.figures_path

    with open(pkl_to_process, 'rb') as pickle_file:
        results_dict = pickle.load(pickle_file)

    if figures_path is not None:
        create_cv_plots(experiment, unit, results_dict, Path(figures_path))
        create_coeff_plots(experiment, unit, results_dict, Path(figures_path))

    if output_path is not None:
        metrics_df = create_metrics_df(results_dict)
        metrics_df.to_csv(Path(output_path, f"{experiment}_{unit}_metrics.csv"), index=False)

        predictions_df = create_predictions_df(results_dict)
        predictions_df.to_csv(Path(output_path, f"{experiment}_{unit}_predictions.csv"), index=False)

        output_model_coeffs(experiment, results_dict, output_path)

    if importance:
        get_feature_importance(results_dict)




