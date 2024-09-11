import csv
import pickle

import pandas as pd

from example_01_training_logistic import *


def get_uuids(fold_dir):
    """
    The function gets the directory where the folds text files are located
    and returns a list of five np.arrays. In each array, the uuids of the
    corresponding fold are stored. Note: This is a modified version of
    get_folds_uuids function.
    """
    # test_uuids_list is a list of np.arrays. Each array is a set of uuids
    test_uuids_list = [0, 1, 2, 3, 4]
    train_uuids_list = [0, 1, 2, 3, 4]
    num_folds = len(test_uuids_list)

    # This loop reads all 5 test folds (iphone and android) and stores uuids
    for i in range(0, num_folds):
        filename = "fold_{}_test_android_uuids.txt".format(i)
        filepath = os.path.join(fold_dir, filename)
        # test1 is the uuids of ith test fold for "android"
        test1 = pd.read_csv(filepath, header=None)
        test1 = test1.values

        filename = "fold_%s_test_iphone_uuids.txt" % i
        filepath = os.path.join(fold_dir, filename)
        # test2 is the uuids of ith test fold for "iphone"
        test2 = pd.read_csv(filepath, header=None)
        test2 = test2.values

        # Then we concatenate them
        test_uuids_list[i] = np.concatenate((test1, test2), axis=0)

        filename = "fold_{}_train_android_uuids.txt".format(i)
        filepath = os.path.join(fold_dir, filename)
        train1 = pd.read_csv(filepath, header=None)
        train1 = train1.values

        filename = "fold_%s_train_iphone_uuids.txt" % i
        filepath = os.path.join(fold_dir, filename)
        train2 = pd.read_csv(filepath, header=None)
        train2 = train2.values

        # Then we concatenate them
        test_uuids_list[i] = np.concatenate((test1, test2), axis=0).flatten()
        train_uuids_list[i] = np.concatenate((train1, train2), axis=0).flatten()

    return train_uuids_list, test_uuids_list


def read_user_data_uncompressed(uuid, data_dir):
    """
    Read the data (precomputed sensor-features and labels) for a user.
    This function assumes the user's data file is present. Note: This
    is a modified version of read_user_data function to read
    uncompressed csv files.
    """
    user_data_file = os.path.join(data_dir, "%s.features_labels.csv" % uuid)

    # Read the entire csv file of the user:
    with open(user_data_file, "r") as fid:
        csv_str = fid.read()
        pass

    (feature_names, label_names) = parse_header_of_csv(csv_str)
    n_features = len(feature_names)
    (X, Y, M, timestamps) = parse_body_of_csv(csv_str, n_features)

    return (X, Y, M, timestamps, feature_names, label_names)


def read_fold_data(uuids, data_dir):
    all_x = []
    all_y = []
    all_m = []
    all_timestamps = []

    for i_uuid in range(len(uuids)):
        (X, Y, M, timestamps, feature_names, label_names) = read_user_data_uncompressed(
            uuids[i_uuid], data_dir
        )
        all_x.append(X)
        all_y.append(Y)
        all_m.append(M)
        all_timestamps.append(timestamps)

    return (
        np.concatenate(all_x, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_m, axis=0),
        np.concatenate(all_timestamps, axis=0),
        feature_names,
        label_names,
    )


def test_model_stripped(
    X_test, Y_test, M_test, timestamps, feat_sensor_names, label_names, model
):
    """
    This is a modified version of test_model function in
    example_01_training_logistic.py.
    :param X_test:
    :param Y_test:
    :param M_test:
    :param timestamps:
    :param feat_sensor_names:
    :param label_names:
    :param model:
    :return:
    """
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    X_test = project_features_to_selected_sensors(
        X_test, feat_sensor_names, model["sensors_to_use"]
    )
    print(
        "== Projected the features to %d features from the sensors: %s"
        % (X_test.shape[1], ", ".join(model["sensors_to_use"]))
    )

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test, model["mean_vec"], model["std_vec"])

    # The single target label:
    target_label = model["target_label"]
    label_ind = label_names.index(target_label)
    y = Y_test[:, label_ind]
    missing_label = M_test[:, label_ind]
    existing_label = np.logical_not(missing_label)

    # Select only the examples that are not missing the target label:
    X_test = X_test[existing_label, :]
    y = y[existing_label]
    timestamps = timestamps[existing_label]

    # Do the same treatment for missing features as done to the training data:
    X_test[np.isnan(X_test)] = 0.0

    print(
        "== Testing with %d examples. For label '%s' we have %d positive and %d negative examples."
        % (len(y), get_label_pretty_name(target_label), sum(y), sum(np.logical_not(y)))
    )

    # Preform the prediction:
    y_pred = model["lr_model"].predict(X_test)

    # Naive accuracy (correct classification rate):
    accuracy = np.mean(y_pred == y)

    # Count occurrences of true-positive, true-negative, false-positive, and false-negative:
    tp = np.sum(np.logical_and(y_pred, y))
    tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y)))
    fp = np.sum(np.logical_and(y_pred, np.logical_not(y)))
    fn = np.sum(np.logical_and(np.logical_not(y_pred), y))

    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)

    # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.0

    # Precision:
    # Beware of this metric, since it may be too sensitive to rare labels.
    # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,
    # and for each label the pos/neg ratio is different.
    # This can cause undesirable and misleading results when averaging precision across different labels.
    precision = float(tp) / (tp + fp)

    print("-" * 10)
    print("Accuracy*:         %.2f" % accuracy)
    print("Sensitivity (TPR): %.2f" % sensitivity)
    print("Specificity (TNR): %.2f" % specificity)
    print("Balanced accuracy: %.2f" % balanced_accuracy)
    print("Precision**:       %.2f" % precision)
    print("-" * 10)

    print(
        "* The accuracy metric is misleading - it is dominated by the negative examples (typically there are many more negatives)."
    )
    print(
        "** Precision is very sensitive to rare labels. It can cause misleading results when averaging precision over different labels."
    )

    # Save results (data and performance metrics) in a dictionary
    results = {
        "y": y,
        "y_pred": y_pred,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
    }

    return results


def main():
    with open(os.path.join("..", "config.toml"), "rb") as f:
        project_config = tomllib.load(f)

    # datadir_compressed = project_config["data_compressed_dir"]
    datadir = project_config["data_dir"]
    folds_dir = project_config["cross_validation_dir"]
    results_dir = os.path.join(os.getcwd(), "results", "02")
    model_file_name_convention = "fold_{}_model.pkl"

    sensors_to_use = ["Acc", "WAcc"]
    target_label = "FIX_walking"

    (train_uuid_list, test_uuid_list) = get_uuids(folds_dir)
    num_folds = len(train_uuid_list)

    # See https://scikit-learn.org/stable/modules/cross_validation.html
    # for help on cross-validation
    for i_fold in range(num_folds):
        print("====================\n== Working on Fold {}".format(i_fold))
        i_train_uuids = train_uuid_list[i_fold]
        i_test_uuids = test_uuid_list[i_fold]
        i_model_file = os.path.join(
            results_dir, "models", model_file_name_convention.format(i_fold)
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
            ) = read_fold_data(i_train_uuids, datadir)

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
        ) = read_fold_data(i_test_uuids, datadir)

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
            results_dir, "fold_{}_metrics.csv".format(i_fold)
        )
        with open(metrics_filename, "w") as f:  # You will need 'wb' mode in Python 2.x
            w = csv.DictWriter(f, results_metrics_only.keys())
            w.writeheader()
            w.writerow(results_metrics_only)

        # Save all results (including true and predicted labels) in pickle file
        # NOTE: This file can be used to generate other metrics that may have been missed
        predictions_filename = os.path.join(
            results_dir, "fold_{}_predictions.pkl".format(i_fold)
        )
        with open(predictions_filename, "wb") as f:
            pickle.dump(results, f)

        # Load results pickle file
        # with open(predictions_filename, "rb") as f:
        #     results_loaded = pickle.load(f)

        print("")


if __name__ == "__main__":
    main()
