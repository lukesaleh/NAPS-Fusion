import csv
import glob
import pickle

from pathlib import Path

import pandas as pd

from example_02_cross_validation import *


def run_cross_val(
    sensors_list,
    activities_list,
    folds_dir,
    data_dir,
    results_dir,
):
    """
    Run cross validation across multiple activities and sensor combinations.
    :param sensors_list:
    :param activities_list:
    :param folds_dir:
    :param data_dir:
    :param results_dir:
    :return:
    """
    (train_uuid_list, test_uuid_list) = get_uuids(folds_dir)

    model_file_name_convention = "fold_{}_model.pkl"

    num_folds = len(train_uuid_list)

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    model_location = {}
    model_num = 1

    for target_label in activities_list:
        for sensors_to_use_unsorted in sensors_list:
            sensors_to_use = sorted(sensors_to_use_unsorted)

            print(
                "====================\n== Activity: {}\n   Sensors (alphabetized): {}\n".format(
                    target_label, ", ".join(sensors_to_use)
                )
            )

            # Save location (relative to results dir) of model
            model_location[model_num] = "{}_{}".format(
                target_label.lower(), "_".join(sensors_to_use).lower()
            )
            fold_results_dir = os.path.join(results_dir, model_location[model_num])

            # Create the models dir from the start
            Path(os.path.join(fold_results_dir, "models")).mkdir(
                parents=True, exist_ok=True
            )

            # See https://scikit-learn.org/stable/modules/cross_validation.html
            # for help on cross-validation
            for i_fold in range(num_folds):
                print("== Working on Fold {}".format(i_fold))
                i_train_uuids = train_uuid_list[i_fold]
                i_test_uuids = test_uuid_list[i_fold]
                i_model_file = os.path.join(
                    fold_results_dir,
                    "models",
                    model_file_name_convention.format(i_fold),
                )

                if os.path.isfile(i_model_file):
                    # Load model
                    with open(i_model_file, "rb") as f:
                        i_model = pickle.load(f)
                else:
                    # Read training data
                    (
                        train_x,
                        train_y,
                        train_missing_labels,
                        train_timestamps,
                        feature_names,
                        label_names,
                    ) = read_fold_data(i_train_uuids, data_dir)

                    feature_sensor_names = get_sensor_names_from_features(feature_names)

                    i_model = train_model(
                        train_x,
                        train_y,
                        train_missing_labels,
                        feature_sensor_names,
                        label_names,
                        sensors_to_use,
                        target_label,
                    )

                    # Save model. See https://stackoverflow.com/questions/66271284/saving-and-reloading-variables-in-python-preserving-names
                    # for helper function
                    with open(i_model_file, "wb") as f:
                        pickle.dump(i_model, f)

                # Read test data
                (
                    test_x,
                    test_y,
                    test_missing_labels,
                    test_timestamps,
                    feature_names,
                    label_names,
                ) = read_fold_data(i_test_uuids, data_dir)

                feature_sensor_names = get_sensor_names_from_features(feature_names)

                results = test_model_stripped(
                    test_x,
                    test_y,
                    test_missing_labels,
                    test_timestamps,
                    feature_sensor_names,
                    label_names,
                    i_model,
                )
                results_metrics_only = results.copy()
                for key in ["y", "y_pred"]:
                    del results_metrics_only[key]

                # Save metrics in csv file
                metrics_filename = os.path.join(
                    fold_results_dir, "fold_{}_metrics.csv".format(i_fold)
                )
                with open(
                    metrics_filename, "w"
                ) as f:  # You will need 'wb' mode in Python 2.x
                    w = csv.DictWriter(f, results_metrics_only.keys())
                    w.writeheader()
                    w.writerow(results_metrics_only)

                # Save all results (including true and predicted labels) in pickle file
                # NOTE: This file can be used to generate other metrics that may have been missed
                predictions_filename = os.path.join(
                    fold_results_dir, "fold_{}_predictions.pkl".format(i_fold)
                )
                with open(predictions_filename, "wb") as f:
                    pickle.dump(results, f)

                # Load results pickle file
                # with open(predictions_filename, "rb") as f:
                #     results_loaded = pickle.load(f)

                print("")

            model_num += 1

            print("")

    # Save model location data in csv file
    save_dict_as_csv(
        os.path.join(results_dir, "location_of_models.csv"), model_location
    )


def evaluate_performance(results_dir, activities_list):
    """
    Evaluate the performance of early fusion for each activity and sensor combination.
    :param results_dir:
    :param activities_list:
    :return:
    """
    # Read model location csv file
    model_location_df = pd.read_csv(os.path.join(results_dir, "location_of_models.csv"))

    model_location = list(model_location_df.values.flatten())

    perf_metrics = [
        "accuracy",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "precision",
    ]
    perf_metrics_num = len(perf_metrics)

    list_of_dfs = []

    # Load data
    for loc in model_location:
        matching_files = glob.glob(
            os.path.join(results_dir, loc, "fold_*[0-9]_metrics.csv")
        )

        metrics_df_list = []
        num_folds = len(matching_files)

        for k_fold in range(num_folds):
            k_df = pd.read_csv(matching_files[k_fold])
            metrics_df_list.append(k_df)

        # Concatenate dataframes
        metrics_df = pd.concat(metrics_df_list, axis=0, ignore_index=True)
        metrics_df["total_samples"] = metrics_df[["tp", "tn", "fp", "fn"]].sum(axis=1)

        list_of_dfs.append(metrics_df)

    return list_of_dfs


def save_dict_as_csv(filename, d):
    with open(filename, "w") as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, d.keys())
        w.writeheader()
        w.writerow(d)


def main():
    with open(os.path.join("..", "config.toml"), "rb") as f:
        project_config = tomllib.load(f)

    data_dir = project_config["data_dir"]
    folds_dir = project_config["cross_validation_dir"]
    results_dir = os.path.join(os.getcwd(), "results", "03")

    sensors_to_use = [["Acc", "WAcc"], ["Gyro", "WAcc"]]

    target_label = ["LYING_DOWN", "FIX_walking", "SITTING", "FIX_running"]

    # Generate results if they do not exist
    if not os.path.isfile(os.path.join(results_dir, "location_of_models.csv")):
        run_cross_val(
            sensors_to_use,
            target_label,
            folds_dir,
            data_dir,
            results_dir,
        )

    # Evaluate performance
    perf_dfs = evaluate_performance(results_dir, target_label)
    df = perf_dfs[2]
    z = 0


if __name__ == "__main__":
    main()
