import sys
import time

from itertools import product
from pycm import ConfusionMatrix
from sklearn.metrics import confusion_matrix

from example_03_early_fusion import *
from not_my_code import get_performance_metrics, plot_confusion_matrix


def get_model_names(activities_list, sensors_list):
    # Alphabetize list of sensors
    sensors_list_alphabetized = [sorted(l) for l in sensors_list]

    return [
        "{}_{}".format(activity.lower(), "_".join(sensors).lower())
        for (activity, sensors) in product(activities_list, sensors_list_alphabetized)
    ]


def save_model_to_pickle(filename, model):
    # Save model. See
    # https://stackoverflow.com/questions/66271284/saving-and-reloading-variables-in-python-preserving-names
    # for helper function
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model_from_pickle(filename):
    # Load model
    with open(filename, "rb") as f:
        model = pickle.load(f)

    return model


def run_cross_val(
    sensors_list,
    activities_list,
    folds_dir,
    data_dir,
    results_dir,
    fusion_method,
    force_test=False,
    force_training=False,
):
    """
    Run cross validation across multiple activities and sensor combinations.
    :param sensors_list:
    :param activities_list:
    :param folds_dir:
    :param data_dir:
    :param results_dir:
    :param fusion_method:
    :param force_test: Re-run model test and overwrite results if they exist
    :param force_training: Force model training even if it exists
    :return:
    """
    (train_uuid_list, test_uuid_list) = get_uuids(folds_dir)

    model_file_name_convention = "fold_{}_model.pkl"

    num_splits = len(train_uuid_list)

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    loc_filename = os.path.join(results_dir, "location_of_models.csv")
    model_location = []

    # Load location_of_models.csv, if it exists
    if os.path.isfile(loc_filename) and (not force_test):
        # TODO: Check for models that no longer exist and remove from list
        model_location = list(pd.read_csv(loc_filename)["folder"])

    # Loop through all activities (a.k.a., target labels)
    for target_label in activities_list:
        # Create a clean list to stow model dictionaries
        fold_models_dict = {}
        for i in range(num_splits):
            fold_models_dict[i] = []

        sensor_model_names = []

        # Train each sensor-activity combination model for all folds
        for sensors_to_use_unsorted in sensors_list:
            sensors_to_use = sorted(sensors_to_use_unsorted)

            print(
                "====================\n== Activity: {}\n   Sensors (alphabetized): {}\n".format(
                    target_label, ", ".join(sensors_to_use)
                )
            )

            # Save location (relative to results dir) of model
            model_name = "{}_{}".format(
                target_label.lower(), "_".join(sensors_to_use).lower()
            )
            sensor_model_names.append(model_name)

            # Add model name to model location list
            if model_name not in model_location:
                model_location.append(model_name)

            fold_results_dir = os.path.join(results_dir, model_name)

            # Create the models dir from the start
            Path(os.path.join(fold_results_dir, "models")).mkdir(
                parents=True, exist_ok=True
            )

            # See https://scikit-learn.org/stable/modules/cross_validation.html
            # for help on cross-validation
            for i_split in range(num_splits):
                print("== Training on all folds except Fold {}".format(i_split))
                i_train_uuids = train_uuid_list[i_split]
                # i_test_uuids = test_uuid_list[i_split]
                i_model_file = os.path.join(
                    fold_results_dir,
                    "models",
                    model_file_name_convention.format(i_split),
                )

                if os.path.isfile(i_model_file) and (not force_training):
                    # Load model if it exists and were are not forcing the training
                    i_model = load_model_from_pickle(i_model_file)

                    print("   - (Skipping training, model already exists)")
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

                    # Train model
                    i_model = train_model(
                        train_x,
                        train_y,
                        train_missing_labels,
                        feature_sensor_names,
                        label_names,
                        sensors_to_use,
                        target_label,
                    )

                    # Save model
                    save_model_to_pickle(i_model_file, i_model)

                fold_models_dict[i_split].append(i_model)

            print("")

        print("\n====================")

        # Test the fusion method on every fold
        for i_split in range(num_splits):
            print("== Testing on Fold {}".format(i_split))
            i_test_uuids = test_uuid_list[i_split]

            fusion_results_dir = os.path.join(
                results_dir, "results_" + fusion_method, target_label.lower()
            )

            # Check if file exists
            if os.path.isfile(
                os.path.join(fusion_results_dir, f"fold_{num_splits-1}_predictions.pkl")
            ) and (not force_test):
                print("   - (Skipping testing, results already exist)")
                continue

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

            i_fold_test_results = test_model(
                test_x,
                test_y,
                test_missing_labels,
                test_timestamps,
                feature_sensor_names,
                label_names,
                fold_models_dict[i_split],
                activities_list,
            )

            # Check that the same labels (i.e., samples) were used across all sensors
            num_sensors = len(fold_models_dict[i_split])
            all_true_labels = [i_fold_test_results[i]["y"] for i in range(num_sensors)]
            all_true_labels = np.vstack(all_true_labels)
            assert np.unique(all_true_labels, axis=0).shape[0] == 1

            # Check that the same index mask was used across all sensors
            all_index_masks = [
                i_fold_test_results[i]["y_keep_index_mask"] for i in range(num_sensors)
            ]
            all_index_masks = np.vstack(all_index_masks)
            assert np.unique(all_index_masks, axis=0).shape[0] == 1

            # Save the individual model performance results of each sensor
            # model, not the fusion method performance
            for j_sensor in range(num_sensors):
                # Find the name of model folder
                fold_results_dir = os.path.join(
                    results_dir, sensor_model_names[j_sensor]
                )

                perf_save(i_fold_test_results[j_sensor], fold_results_dir, i_split)

                # results_metrics_only = i_fold_test_results[j_sensor].copy()
                # for key in ["y", "y_pred", "y_prob", "index_mask"]:
                #     del results_metrics_only[key]
                #
                # # Save metrics in csv file
                # metrics_filename = os.path.join(
                #     fold_results_dir, "fold_{}_metrics.csv".format(i_split)
                # )
                # # You will need 'wb' mode in Python 2.x
                # with open(metrics_filename, "w") as f:
                #     w = csv.DictWriter(f, results_metrics_only.keys())
                #     w.writeheader()
                #     w.writerow(results_metrics_only)
                #
                # # Save all results (including true and predicted labels) in pickle file
                # # NOTE: This file can be used to generate other metrics that may have been missed
                # predictions_filename = os.path.join(
                #     fold_results_dir, "fold_{}_predictions.pkl".format(i_split)
                # )
                # with open(predictions_filename, "wb") as f:
                #     pickle.dump(i_fold_test_results[j_sensor], f)
                #
                # # NOTE: For debugging. Load results pickle file
                # # with open(predictions_filename, "rb") as f:
                # #     results_loaded = pickle.load(f)

            # Calculate fusion performance results depending on the method
            if fusion_method == "lfa-fusion":
                i_sensor_test_probabilities = [
                    i_fold_test_results[jj]["y_prob"] for jj in range(num_sensors)
                ]

                # Place all sensor probabilities into a single array
                i_sensor_test_probabilities = np.vstack(i_sensor_test_probabilities)

                # Calculate the mean probability of an activity using
                # probabilities of all sensors
                i_fusion_prob = i_sensor_test_probabilities.mean(axis=0)

                # Calculate the final predictions
                i_fusion_pred = np.zeros(i_fusion_prob.shape, dtype=bool)
                i_fusion_pred[i_fusion_prob > 0.5] = True
            elif fusion_method == "lfl-fusion":
                i_lfl_model_file = os.path.join(
                    fusion_results_dir, "models", f"fold_{i_split}_model.pkl"
                )
                Path(os.path.dirname(i_lfl_model_file)).mkdir(
                    parents=True, exist_ok=True
                )

                if os.path.isfile(i_lfl_model_file) and (not force_training):
                    i_lfl_model = load_model_from_pickle(i_lfl_model_file)
                    i_lfl_lr_model = i_lfl_model["lr_model"]
                else:
                    # Load training sample probabilities from each sensor model dictionary
                    i_sensor_train_probabilities = [
                        fold_models_dict[i_split][jj]["y_prob"]
                        for jj in range(num_sensors)
                    ]
                    i_sensor_train_probabilities = np.vstack(
                        i_sensor_train_probabilities
                    ).T

                    i_sensor_train_labels = [
                        fold_models_dict[i_split][jj]["y"] for jj in range(num_sensors)
                    ]
                    i_sensor_train_labels = np.vstack(i_sensor_train_labels)

                    # Check that test labels for all sensors are the same
                    assert np.unique(i_sensor_train_labels, axis=0).shape[0] == 1
                    i_sensor_train_labels = i_sensor_train_labels[0, :]

                    # Train a binary LR classifier using training sample probabilities
                    i_lfl_lr_model = sklearn.linear_model.LogisticRegression(
                        class_weight="balanced", max_iter=1000
                    )
                    i_lfl_lr_model.fit(
                        i_sensor_train_probabilities, i_sensor_train_labels
                    )

                    i_lfl_model = {
                        "sensors_to_use": sensors_list,
                        "target_label": target_label,
                        "lr_model": i_lfl_lr_model,
                        "y": i_sensor_train_labels,
                        "y_prob": i_sensor_train_probabilities,
                    }

                    # Save model
                    save_model_to_pickle(i_lfl_model_file, i_lfl_model)

                # Test on test sample probabilities
                i_sensor_test_probabilities = [
                    i_fold_test_results[jj]["y_prob"] for jj in range(num_sensors)
                ]
                i_sensor_test_probabilities = np.vstack(i_sensor_test_probabilities).T
                i_fusion_pred = i_lfl_lr_model.predict(i_sensor_test_probabilities)
                i_fusion_prob = i_lfl_lr_model.predict_proba(
                    i_sensor_test_probabilities
                )[:, 1]
            else:
                print(
                    "Chose fusion method: '{}' has not been implemented".format(
                        fusion_method
                    )
                )
                sys.exit(1)

            # Any y_prob can be used as they should all be the same, due to
            # earlier assertion
            i_fusion_true_labels = i_fold_test_results[0]["y"]

            # Obtain performance metrics
            i_perf_metrics = perf_measure(i_fusion_true_labels, i_fusion_pred)
            i_fusion_results = {
                "y_all": i_fold_test_results[0]["y_all"],
                "y_all_label_idx": i_fold_test_results[0]["y_all_label_idx"],
                "y_target_label": i_fold_test_results[0]["y_target_label"],
                "y": i_fusion_true_labels,
                "y_pred": i_fusion_pred,
                "y_prob": i_fusion_prob,
                "tp": i_perf_metrics["tp"],
                "tn": i_perf_metrics["tn"],
                "fp": i_perf_metrics["fp"],
                "fn": i_perf_metrics["fn"],
                "accuracy": i_perf_metrics["accuracy"],
                "sensitivity": i_perf_metrics["sensitivity"],
                "specificity": i_perf_metrics["specificity"],
                "balanced_accuracy": i_perf_metrics["balanced_accuracy"],
                "precision": i_perf_metrics["precision"],
            }

            # Save fusion predictions and performance metrics
            perf_save(
                i_fusion_results,
                fusion_results_dir,
                i_split,
            )

            print("")

    # Save model location data in csv file
    model_location_dict = {
        "folder": model_location,
    }
    model_location_df = pd.DataFrame(model_location_dict)
    model_location_df.to_csv(loc_filename, index=False)

    print("")


def summarize_performance(
    results_dir,
    activities_list,
):
    """
    Evaluate the performance of early fusion for each activity and sensor combination.
    :param results_dir: Fusion results directory
    :param activities_list:
    :return:
    """
    # TODO: Remember order that labels were saved in results and allow user
    #  to specify activities in different order to plot CM differently
    activities_lowercase = [label.lower() for label in activities_list]
    num_labels = len(activities_lowercase)

    # NOTE: From here on forward, a label refers to the number assigned to a
    # class
    label_names = {i: label for (i, label) in enumerate(activities_lowercase)}
    label_pretty_names = {
        i: get_label_pretty_name(label)
        for (i, label) in enumerate(activities_lowercase)
    }

    # Find the number of folds by looking for the number of pickle files in the
    # last activity's folder
    pkl_files_list = glob.glob(
        os.path.join(
            os.path.join(results_dir, activities_lowercase[-1]),
            "fold_*_predictions.pkl",
        ),
        recursive=False,
    )
    num_folds = len(pkl_files_list)
    assert num_folds > 0

    multi_class_results = []

    for i_fold in range(num_folds):
        i_fold_results = [
            load_model_from_pickle(
                os.path.join(results_dir, label, f"fold_{i_fold}_predictions.pkl")
            )
            for label in activities_lowercase
        ]

        i_fold_true_y = [i_fold_results[jj]["y_all"] for jj in range(num_labels)]
        i_fold_true_y = np.vstack(i_fold_true_y)
        # Indirectly check that we used the same sample indices across all
        # activities
        assert np.unique(i_fold_true_y, axis=0).shape[0] == 1
        i_fold_true_y = i_fold_true_y[0, :]

        i_fold_probs = [i_fold_results[jj]["y_prob"] for jj in range(num_labels)]
        i_fold_probs = np.vstack(i_fold_probs)

        i_fold_predicted_y = np.argmax(i_fold_probs, axis=0)

        # Append results to list
        multi_class_results.append(
            {
                "true_class": i_fold_true_y,
                "predicted_class": i_fold_predicted_y,
                "label_names": label_names,
                "label_pretty_names": label_pretty_names,
            }
        )

    # Save results
    save_model_to_pickle(
        os.path.join(results_dir, "multiclass_results.pkl"), multi_class_results
    )

    all_true_y = np.concatenate(
        [multi_class_results[ii]["true_class"] for ii in range(num_folds)]
    )
    all_predicted_y = np.concatenate(
        [multi_class_results[ii]["predicted_class"] for ii in range(num_folds)]
    )

    # Calculate performance metric using PyCM
    perf_metrics_pycm = ConfusionMatrix(
        actual_vector=all_true_y, predict_vector=all_predicted_y
    )
    perf_metrics_pycm.relabel(mapping=label_pretty_names)
    print(perf_metrics_pycm)

    # Experimental plots
    # perf_metrics_pycm.plot(cmap=plt.cm.Blues, number_label=True, plot_lib="matplotlib")
    # perf_metrics_pycm.plot(
    #     cmap=plt.cm.Blues, number_label=True, plot_lib="seaborn", normalized=True
    # )

    # Calculate performance metrics (stored in dict) using code found online.
    # This shows balanced accuracy, which PyCM does not calculated.
    conf_mat = confusion_matrix(all_true_y, all_predicted_y)
    perf_metrics = get_performance_metrics(conf_mat)
    print("Balanced accuracy =", perf_metrics["balanced_accuracy"])

    plot_confusion_matrix(
        all_true_y,
        all_predicted_y,
        list(label_pretty_names.values()),
        os.path.join(results_dir, "figures"),
    )

    perf_metrics_pycm.save_csv(
        os.path.join(results_dir, "performance_metrics"),
        matrix_save=True,
        # Add class labels to confusion matrix
        header=True,
    )

    print("\n")


# NOTE: ovr is selected for multi_class is data is binary. What is binary data?
# NOTE: To answer above, adding multi_class="multinomial" may not make a difference
def train_model(
    X_train,
    Y_train,
    M_train,
    feat_sensor_names,
    label_names,
    sensors_to_use,
    target_label,
):
    # Project the feature matrix to the features from the desired sensors:
    X_train = project_features_to_selected_sensors(
        X_train, feat_sensor_names, sensors_to_use
    )
    print(
        "== Projected the features to %d features from the sensors: %s"
        % (X_train.shape[1], ", ".join(sensors_to_use))
    )

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    (mean_vec, std_vec) = estimate_standardization_params(X_train)
    X_train = standardize_features(X_train, mean_vec, std_vec)

    # The single target label:
    # NOTE: Original code is not case-insensitive
    label_names_lowercase = [label.lower() for label in label_names]
    label_ind = label_names_lowercase.index(target_label.lower())
    y = Y_train[:, label_ind]
    missing_label = M_train[:, label_ind]
    existing_label = np.logical_not(missing_label)

    # Select only the examples that are not missing the target label:
    X_train = X_train[existing_label, :]
    y = y[existing_label]

    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
    # You can also further select examples - only those that have values for all the features.
    # For this tutorial, let's use the simple heuristic of zero-imputation:
    X_train[np.isnan(X_train)] = 0.0

    print(
        "== Training with %d examples. For label '%s' we have %d positive and %d negative examples."
        % (len(y), get_label_pretty_name(target_label), sum(y), sum(np.logical_not(y)))
    )

    # Now, we have the input features and the ground truth for the output label.
    # We can train a logistic regression model.

    # Typically, the data is highly imbalanced, with many more negative examples;
    # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
    lr_model = sklearn.linear_model.LogisticRegression(
        class_weight="balanced", max_iter=1000
    )
    lr_model.fit(X_train, y)

    # Prediction probability of the training data
    train_prob = lr_model.predict_proba(X_train)[:, 1]

    # Assemble all the parts of the model:
    model = {
        "sensors_to_use": sensors_to_use,
        "target_label": target_label,
        "mean_vec": mean_vec,
        "std_vec": std_vec,
        "lr_model": lr_model,
        "y": y,
        "y_prob": train_prob,
    }

    return model


def test_model(
    X_test,
    Y_test,
    M_test,
    timestamps,
    feat_sensor_names,
    label_names,
    model_list,
    all_activities,
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
    :param model_list:
    :all_activities:
    :return:
    """
    num_sensors = len(model_list)
    all_results = []

    # Loop through all sensors to obtain the probability that the subject is
    # walking giving a sensor's readings
    # NOTE: The (all activities, all folds) loops are in the outer loops that call this function
    for i_sensor in range(num_sensors):
        i_model = model_list[i_sensor]

        # Project the feature matrix to the features from the sensors that the classifier is based on:
        i_x_test = project_features_to_selected_sensors(
            X_test, feat_sensor_names, i_model["sensors_to_use"]
        )
        print(
            "  - Projected the features to %d features from the sensors: %s"
            % (i_x_test.shape[1], ", ".join(i_model["sensors_to_use"]))
        )

        # We should standardize the features the same way the train data was standardized:
        i_x_test = standardize_features(
            i_x_test, i_model["mean_vec"], i_model["std_vec"]
        )

        # The single target label:
        target_label = i_model["target_label"]

        # Find class labels (positive class is assigned True and negative
        # class is assigned False)
        # NOTE: Original code is not case-insensitive
        label_names_lowercase = [label.lower() for label in label_names]
        label_ind = label_names_lowercase.index(target_label.lower())
        y = Y_test[:, label_ind].copy()

        # Select only the examples that are not missing the target label
        # (old way)
        # missing_label = M_test[:, label_ind].copy()
        # existing_label = np.logical_not(missing_label)
        # i_x_test = i_x_test[existing_label, :]
        # y = y[existing_label]
        # # i_timestamps = timestamps[existing_label].copy()

        # Select only the examples that are not missing the target label
        # (new way). This also finds detailed multi-class labels of test
        # samples
        # NOTE: Ideally, some of this code, if not all, should be done in
        # read_fold_data() or read_user_data_uncompressed()
        all_target_labels = [a.lower() for a in all_activities]
        all_label_idx = [label_names_lowercase.index(l) for l in all_target_labels]
        y_all_binary = Y_test[:, all_label_idx].copy()

        # Check that no more than 1 label is assigned to each sample
        # NOTE: Some samples may not be in any of the chosen classes
        num_classes_per_sample = y_all_binary.sum(axis=1)
        assert np.array_equal(np.unique(num_classes_per_sample), [0, 1])

        # Keep "good" test data
        existing_label = num_classes_per_sample == 1
        i_x_test = i_x_test[existing_label, :]
        y = y[existing_label]
        y_all_binary = y_all_binary[existing_label, :]
        num_test_samples = y_all_binary.shape[0]

        # -1 means none of the classes
        y_all = -1 * np.ones(num_test_samples, dtype=int)

        for i_col in range(len(all_label_idx)):
            y_all[y_all_binary[:, i_col]] = i_col

        # assert num_classes_per_sample.sum() == (y_all >= 0).sum()
        assert num_test_samples == (y_all >= 0).sum()  # stronger check

        # Do the same treatment for missing features as done to the training
        # data (see train_model function)
        i_x_test[np.isnan(i_x_test)] = 0.0

        print(
            "   - Testing with %d examples. For label '%s' we have %d positive and %d negative examples."
            % (
                len(y),
                get_label_pretty_name(target_label),
                sum(y),
                sum(np.logical_not(y)),
            )
        )

        # Perform the prediction. True is given when probability of positive class > 0.5
        y_pred = i_model["lr_model"].predict(i_x_test)

        # Find the probability of the positive class
        y_prob = i_model["lr_model"].predict_proba(i_x_test)[:, 1]

        perf_metrics = perf_measure(y, y_pred)

        print("  ", "-" * 10)
        print("   Accuracy*:         %.2f" % perf_metrics["accuracy"])
        print("   Sensitivity (TPR): %.2f" % perf_metrics["sensitivity"])
        print("   Specificity (TNR): %.2f" % perf_metrics["specificity"])
        print("   Balanced accuracy: %.2f" % perf_metrics["balanced_accuracy"])
        print("   Precision**:       %.2f" % perf_metrics["precision"])
        print("  ", "-" * 10)

        # print(
        #     "   * The accuracy metric is misleading - it is dominated by the negative examples (typically there are many more negatives)."
        # )
        # print(
        #     "   ** Precision is very sensitive to rare labels. It can cause misleading results when averaging precision over different labels."
        # )

        # Save results (data and performance metrics) in a dictionary
        results = {
            "y_all": y_all,
            "y_all_label_idx": all_target_labels.index(target_label.lower()),
            "y_target_label": target_label.lower(),
            "y": y,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_keep_index_mask": existing_label,
            "sensors_used": i_model["sensors_to_use"],
            "tp": perf_metrics["tp"],
            "tn": perf_metrics["tn"],
            "fp": perf_metrics["fp"],
            "fn": perf_metrics["fn"],
            "accuracy": perf_metrics["accuracy"],
            "sensitivity": perf_metrics["sensitivity"],
            "specificity": perf_metrics["specificity"],
            "balanced_accuracy": perf_metrics["balanced_accuracy"],
            "precision": perf_metrics["precision"],
        }

        all_results.append(results)

    return all_results


def perf_measure(y, y_pred):
    """
    Calculate various performance metrics and place into dictionary
    :param y:
    :param y_pred:
    :return:
    """
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

    return {
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


def perf_save(results_metrics: dict, save_dir: str, split_num: int):
    csv_incompatible_keys = [
        "y_all",
        "y_all_label_idx",
        "y_target_label",
        "y",
        "y_pred",
        "y_prob",
        "y_keep_index_mask",
        "sensors_used",
    ]
    incompatible_keys_found = [
        k for k in results_metrics.keys() if k in csv_incompatible_keys
    ]

    results_metrics_only = results_metrics.copy()

    for key in incompatible_keys_found:
        del results_metrics_only[key]

    # Save metrics in csv file
    metrics_filename = os.path.join(save_dir, "fold_{}_metrics.csv".format(split_num))

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # You will need 'wb' mode in Python 2.x
    with open(metrics_filename, "w") as f:
        w = csv.DictWriter(f, results_metrics_only.keys())
        w.writeheader()
        w.writerow(results_metrics_only)

    # Save all results (including true and predicted labels) in pickle file
    # NOTE: With this pickle file, we can generate other metrics that may have been missed
    predictions_filename = os.path.join(
        save_dir, "fold_{}_predictions.pkl".format(split_num)
    )
    with open(predictions_filename, "wb") as f:
        pickle.dump(results_metrics, f)


def main():
    with open(os.path.join("..", "config.toml"), "rb") as f:
        project_config = tomllib.load(f)

    data_dir = project_config["data_dir"]
    folds_dir = project_config["cross_validation_dir"]
    results_dir = os.path.join(os.getcwd(), "results", "04")

    fusion_method = project_config["fusion_method"]
    sensors_list = project_config[fusion_method]["sensor_combination"]
    target_labels = project_config[fusion_method]["activities"]

    # Generate models and results
    run_cross_val(
        sensors_list,
        target_labels,
        folds_dir,
        data_dir,
        results_dir,
        fusion_method,
        force_test=project_config[fusion_method]["force_test"],
        force_training=project_config[fusion_method]["force_training"],
    )

    # Evaluate performance
    summarize_performance(
        os.path.join(results_dir, "results_" + fusion_method),
        target_labels,
    )


if __name__ == "__main__":
    # See https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions
    t = time.time()
    main()
    elapsed = time.time() - t
    print(
        f"Elapsed time = {elapsed:.1f} s = {int(elapsed // 60)} min {elapsed % 60:.1f} s"
    )
