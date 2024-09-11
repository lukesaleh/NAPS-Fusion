try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np
import pandas as pd

import ExtraSensory_UCSD_LFL as lfl_code

# import ExtraSensory_UCSD_LFA as lfa_code


def test_perf_measure():
    y_actual = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]

    (tp, fp, tn, fn) = lfl_code.perf_measure(y_actual, y_pred)

    if (tp, fp, tn, fn) != (4, 2, 3, 1):
        print("test_perf_measure: failed")


def test_get_folds_uuids(fold_dir):
    list_of_uuids = lfl_code.get_folds_uuids(fold_dir)
    all_uuids = np.concatenate(list_of_uuids[:])
    all_uuids_df = pd.DataFrame(all_uuids)

    # Generate comparison CSV file (only do this once!)
    # all_uuids_df.to_csv("uuids.csv", header=False, index=False)

    comparison_df = pd.read_csv("uuids.csv", header=None)

    if not all_uuids_df.equals(comparison_df):
        print("test_get_folds_uuids: failed")


def test_readdata_csv(data_dir):
    ds_dict = lfl_code.readdata_csv(data_dir)
    num_values = [len(ds_dict[key]) for key in ds_dict]
    if sum(num_values) != 377346:
        print("test_get_folds_uuids: failed")


if __name__ == "__main__":
    with open("../config.toml", "rb") as f:
        project_config = tomllib.load(f)

    datadir = project_config["data_dir"]
    cvdir = project_config["cross_validation_dir"]

    test_perf_measure()
    test_get_folds_uuids(cvdir)
    test_readdata_csv(datadir)
