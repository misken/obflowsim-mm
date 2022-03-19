import argparse
import sys

import pandas as pd


def process_command_line():
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='mm_merge_predict_simulated',
                                     description='merge simulation scenario output with mm predictions')

    # Add arguments
    parser.add_argument(
        "experiment", type=str,
        help="String used in output filenames"
    )

    parser.add_argument(
        "perf_curve_pred_filename", type=str,
        help="Path to csv file which contains predictions"
    )

    parser.add_argument(
        "y_data_path", type=str,
        help="Path to directory containing y data files (which are created from sim output)"
    )

    parser.add_argument(
        "output_filename", type=str,
        help="Path to merged output csv file"
    )

    # Do the parsing and return the populated namespace with the input arg values
    args = parser.parse_args()
    return args


def main(argv=None):
    # Parse command line arguments
    mm_args = process_command_line()
    exp = mm_args.experiment
    perf_curve_pred_filename = mm_args.perf_curve_pred_filename
    y_data_path = mm_args.y_data_path
    output_filename = mm_args.output_filename

    predictions_df = pd.read_csv(perf_curve_pred_filename)

    unit_measure_pairs = [('ldr', 'occmean'), ('ldr', 'occp95'), ('ldr', 'probblocked'), ('ldr', 'condmeantimeblocked'),
                          ('obs', 'occmean'), ('obs', 'occp95'), ('obs', 'probblocked'), ('obs', 'condmeantimeblocked'),
                          ('pp', 'occmean'), ('pp', 'occp95')]

    y_dfs = []
    for (unit, measure) in unit_measure_pairs:
        y_filename = f'{y_data_path}/y_{unit}_{measure}_{exp}.csv'
        y_df = pd.read_csv(y_filename)
        y_df.columns = ['scenario', 'simulated']
        y_df['unit'] = unit
        y_df['measure'] = measure
        y_dfs.append(y_df)

    simulated_df = pd.concat(y_dfs)

    # Merge in actual values with predictions
    predictions_simulated_df = predictions_df.merge(simulated_df, how='left', on=['scenario', 'unit', 'measure'])

    predictions_simulated_df.to_csv(output_filename, index=False)


if __name__ == '__main__':
    sys.exit(main())

