"""
Basic implementation of NAPS Fusion for activity recognition in the publication
EJ, "...", 2023
"""
import copy
import csv
import os
import random
import sys
import time

from pathlib import Path

from imblearn.over_sampling import SMOTE
from pycm import ConfusionMatrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

# import config
import numpy as np
import pyds_local as pyds
from sklearn.model_selection import train_test_split

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, "../examples")
import ucsd_fusion as uf
import not_my_code as nmc
import Naive_Adaptive_Sensor_Fusion as naps


class Timer(object):
    def __init__(self, name=None):
        """
        Nabbed from https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions
        :param name: Name of timer instance
        """
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(
                "[%s]" % self.name,
            )
        print("Elapsed: %s" % (time.time() - self.tstart))


def clean_data(train_x: int, train_y, all_labels, chosen_activities):
    """
    Normalize data & Remove samples with missing labels

    :param train_x:
    :param train_y:
    :param all_labels:
    :param chosen_activities:
    :return:
    """

    # ------------------------------
    # Preprocessing training data
    # NOTE: Some of this code is borrowed from the UCSD fusion training code
    # ------------------------------

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    (mean_vec, std_vec) = uf.estimate_standardization_params(train_x)
    train_x = uf.standardize_features(train_x, mean_vec, std_vec)

    # Handle missing features values (i.e., NaN values) using
    # zero-imputation
    # NOTE: There are other ways to handle this
    train_x[np.isnan(train_x)] = 0.0

    assert np.isnan(train_x).sum() == 0

    # ------------------------------
    # Remove missing samples
    # ------------------------------

    all_labels_lowercase = [label.lower() for label in all_labels]
    chosen_labels_indices = [
        all_labels_lowercase.index(l.lower()) for l in chosen_activities
    ]

    train_y = train_y[:, chosen_labels_indices]
    has_label = np.any(train_y, axis=1)

    # Select only the examples that have an activity label
    train_x = train_x[has_label]
    train_y = train_y[has_label]

    assert np.all(train_y.sum(axis=1) == 1)

    return train_x, train_y


def find_reduced_set_indices(
    num_reduced_sets,
    features_ratio,
    sensor_indices,
    sensors_list,
    rng,
):
    """
    Select random features from a dataset.

    :param num_reduced_sets:
    :param features_ratio:
    :param sensor_indices:
    :param sensors_list:
    :param rng:
    :return:
    """
    reduced_set_indices = []

    for i_set in range(num_reduced_sets):
        # Find each sensor's sampled features
        sampled_features_per_sensor = {}
        sampled_features = np.array([], dtype=int)
        for s in sensors_list:
            num_features = len(sensor_indices[s])
            num_sampled_features = int(features_ratio * num_features)
            s_feat = rng.choice(
                sensor_indices[s], size=num_sampled_features, replace=False
            )
            sampled_features_per_sensor.update({s: s_feat})
            sampled_features = np.append(sampled_features, s_feat)

        reduced_set_indices.append(
            {
                "sampled_features_per_sensor": sampled_features_per_sensor,
                "sampled_features": sampled_features,
            }
        )

    return reduced_set_indices


class DSModel:
    def __init__(
        self, num_bags, resp_var_1, resp_var_2, feature_set_idx, class_uncertainty
    ):
        """
        Initializing a DS model

        :param num_bags: num_bags in bagging classifier
        :type resp_var_1: list
        :param resp_var_1:
        :type resp_var_2: list
        :param resp_var_2: These two are the response variables that we want to build a model
            on. For example if we are building a model to classify {c1} vs {c2,c3}
            then [c1] would be our resp_var_1 and [c2,c3] would be our resp_var_2.
        :type feature_set_idx: int
        :param feature_set_idx: Index of the reduced feature set (out of all the randomly created feature sets)
        :type class_uncertainty: float
        :param class_uncertainty: Uncertainty due to the imbalanced class

        :return:
        """
        self.Feature_Set_idx = feature_set_idx
        self.Response_Variables = [resp_var_1, resp_var_2]
        self.NumBags = num_bags
        self.Uncertainty_B = class_uncertainty  # Uncertainty of the biased model
        self.mass = pyds.MassFunction()  # Mass function of the model
        self.C = len(self.Response_Variables)  # Number of the classes

    def Mass_Function_Setter(self, uncertainty, test_x_prob):
        """
        We used pyds package (a Dempster-Shafer package) to define/initialize the mass function
        given the probabilities and uncertainty.

        :param uncertainty:
        :param test_x_prob:
        :return:
        """
        # Zero-out the masses prior to assignment. This covers the case
        # when DS Model has been used for something else
        for key in self.mass:
            self.mass[key] = 0.0
        # Can I do this? Doesn't look like it
        # self.mass = {}

        self.mass[self.Response_Variables[0]] = test_x_prob[0] * (1 - uncertainty)
        self.mass[self.Response_Variables[1]] = test_x_prob[1] * (1 - uncertainty)
        self.mass[self.Response_Variables[0] + self.Response_Variables[1]] = uncertainty

    def Uncertainty_Context(self, test_x_prob):
        """
        This function calculates the uncertainty of the contextual meaning by
        calculating the votes from all the bags.

        :param test_x_prob: Precalculated class probability of observation
        :return:
        """
        C = len(self.Response_Variables)
        assert len(test_x_prob) == C

        # Vote vector
        V = np.round(test_x_prob * self.NumBags).astype(int)

        uncertainty = 1 - np.sqrt(
            np.sum(np.power((V / self.NumBags - 1 / C), 2))
        ) / np.sqrt((C - 1) / C)

        return uncertainty


def train_model(
    train_x,
    train_y,
    train_missing_labels,
    feat_sensor_names,
    label_names,
    sensors_used,
    activities_list,
    reduced_set_indices,
    class_combo_info,
    fusion_config,
):
    """

    :param train_x:
    :param train_y:
    :param train_missing_labels:
    :param feat_sensor_names:
    :param label_names:
    :param sensors_used:
    :param activities_list:
    :param reduced_set_indices:
    :param class_combo_info:
    :param fusion_config:
    :return:
    """
    # ------------------------------
    # Preprocessing training data
    # NOTE: Some of this code is borrowed from the UCSD fusion training code
    # ------------------------------

    # # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # # so that all their values will be roughly in the same range:
    # (mean_vec, std_vec) = uf.estimate_standardization_params(train_x)
    # train_x = uf.standardize_features(train_x, mean_vec, std_vec)
    #
    # # Handle missing features values (i.e., NaN values) using
    # # zero-imputation
    # # NOTE: There are other ways to handle this
    # train_x[np.isnan(train_x)] = 0.0

    # Find positive and negative augmented classes (i.e., augmented responses)
    mass_template_df = naps.BPA_builder(fusion_config["activities"])
    pos_resp, neg_resp = naps.pair_resp(mass_template_df)

    # ------------------------------
    # Balance out the binary classes using SMOTE
    # ------------------------------
    feature_bagging_ratio = fusion_config["bagging_features_ratio"]
    samples_bagging_ratio = fusion_config["bagging_samples_ratio"]
    bootstrap_features = fusion_config["bootstrap_features"]
    bootstrap_samples = fusion_config["bootstrap_samples"]

    num_reduced_sets = len(reduced_set_indices)
    num_proposition_combos = len(class_combo_info)

    num_bags = fusion_config["number_bags"]
    base_random_seed = fusion_config["random_seed"]
    num_original_samples = train_x.shape[0]

    # # What was I doing here?
    # samples_in_reduced_sets = np.array(
    #     [len(d["sampled_features"]) for d in reduced_set_indices]
    # )
    # assert len(np.unique(samples_in_reduced_sets)) == 1
    # samples_in_reduced_sets = samples_in_reduced_sets[0]
    # # NOTE: This variable is only useful for debugging, for now
    # samples_in_bag = int(
    #     np.round(samples_bagging_ratio * samples_in_reduced_sets)
    # )

    reduced_set_models = []

    for i in range(num_reduced_sets):
        print(f"    - Working on reduced set {i + 1} of {num_reduced_sets}")

        # Create reduced feature set
        reduced_x = train_x[:, reduced_set_indices[i]["sampled_features"]]

        models_per_combination = []

        for j in range(num_proposition_combos):
            # Find the augmented classes
            # NOTE: The positive class is represented by the integer 1,
            # 0 for negative
            reduced_y = np.zeros(num_original_samples, dtype=int)
            reduced_y[class_combo_info[j]["positive_class_mask"]] = 1

            # Quantify model uncertainty due to class imbalance
            # NOTE: This code is ad-hoc for two classes
            class_uncertainty = naps.Uncertainty_Bias(
                [reduced_y, np.abs(reduced_y - 1)]
            )

            # NOTE: n_neighbors is tweakable, see code for SMOTE to guide you
            base_estimator = NearestNeighbors(n_neighbors=6, n_jobs=-1)
            smt = SMOTE(
                random_state=base_random_seed + i, k_neighbors=clone(base_estimator)
            )
            balanced_x, balanced_y = smt.fit_resample(reduced_x, reduced_y)

            base_clf = DecisionTreeClassifier()
            bagging_clf = BaggingClassifier(
                estimator=base_clf,
                n_estimators=num_bags,
                random_state=base_random_seed + i,
                n_jobs=-1,
                max_features=feature_bagging_ratio,
                max_samples=samples_bagging_ratio,
                bootstrap_features=bootstrap_features,
                bootstrap=bootstrap_samples,
            )

            # ------------------------------
            # Train!
            # ------------------------------
            bagging_clf.fit(balanced_x, balanced_y)

            # NOTE: Code to test the performance on training data; useful for
            # debugging
            j_results = bagging_clf.predict(reduced_x)
            j_proba = bagging_clf.predict_proba(reduced_x)
            j_mean_acc = bagging_clf.score(reduced_x, reduced_y)

            # Create DSModel instance
            ij_ds_model = DSModel(
                bagging_clf.n_estimators, pos_resp[j], neg_resp[j], i, class_uncertainty
            )

            models_per_combination.append(
                {
                    "ds_model": ij_ds_model,
                    "bagging_classifier": bagging_clf,
                    "bagging_prediction_class_on_training_data": j_results,
                    "bagging_prediction_probability_on_training_data": j_proba,
                    "bagging_prediction_mean_accuracy_on_training_data": j_mean_acc,
                    "positive_class_indices": class_combo_info[j][
                        "positive_class_indices"
                    ],
                    "positive_class_activities": pos_resp[j],
                    "negative_class_indices": class_combo_info[j][
                        "negative_class_indices"
                    ],
                    "negative_class_activities": neg_resp[j],
                }
            )

        reduced_set_models.append(models_per_combination)

    return reduced_set_models


def run_cross_val(config: dict):
    """
    Run cross validation across multiple activities and sensor combinations.
    :param config: Project config
    :return:
    """
    # ----------
    # Load project configuration
    # ----------
    fusion_method = config["fusion_method"]

    data_dir = config["data_dir"]
    folds_dir = config["cross_validation_dir"]
    results_dir = config[fusion_method]["results_dir"]
    sensors_list = config[fusion_method]["sensor_combination"]
    activities_list = config[fusion_method]["activities"]
    num_activities = len(activities_list)

    force_test = config[fusion_method]["force_test"]
    force_training = config[fusion_method]["force_training"]

    # NOTE: This could be potentially moved to config.py
    sensor_indices = {
        "Acc": list(range(0, 26)),
        "Gyro": list(range(26, 52)),
        "WAcc": list(range(83, 129)),
        "Loc": list(range(138, 155)),
        "Aud": list(range(155, 181)),
        "PS": list(np.append(range(183, 209), range(217, 225))),
    }

    # ----------
    # Load UUID of training and testing folds
    # ----------

    (train_uuid_list, test_uuid_list) = uf.get_uuids(folds_dir)
    num_splits = len(train_uuid_list)

    fold_data_naming_convention = "fold_{}_data.pkl"

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # ----------
    # Select the positive and negative augmented classes
    # ----------

    # # My way
    # # Create data structures for binary class problems
    # num_classes = num_activities
    # class_binary_repr_str = [
    #     np.binary_repr(i, width=num_classes)[::-1]
    #     for i in range(1, 2**num_classes // 2 + 1)
    # ]
    # # Convert strings to list of booleans
    # class_binary_repr_bool_list = []
    # for i_str in class_binary_repr_str:
    #     class_binary_repr_bool_list.append([True if c == "1" else False for c in i_str])
    # class_binary_repr_mask = np.vstack(class_binary_repr_bool_list)
    # keep_row = [
    #     i for i, row in enumerate(class_binary_repr_mask) if row.sum() in [1, 2]
    # ]
    # positive_augmented_class_mask = class_binary_repr_mask[keep_row]
    # positive_augmented_class_idx = [
    #     np.where(m)[0] for m in positive_augmented_class_mask
    # ]
    # num_class_combos = len(positive_augmented_class_idx)

    # Better way
    mass_template_df = naps.BPA_builder(activities_list)
    pos_resp, neg_resp = naps.pair_resp(mass_template_df)

    positive_augmented_class_idx = [
        [activities_list.index(n) for n in m] for m in pos_resp
    ]
    negative_augmented_class_idx = [
        [activities_list.index(n) for n in m] for m in neg_resp
    ]

    # ----------
    # Iterate over fold splits
    # ----------

    chosen_splits = config[fusion_method]["chosen_splits"]

    for i_split in chosen_splits:
        i_train_uuids = train_uuid_list[i_split]
        i_test_uuids = test_uuid_list[i_split]

        i_fold_data_file = os.path.join(
            results_dir,
            fold_data_naming_convention.format(i_split),
        )

        fname, fext = os.path.splitext(i_fold_data_file)
        i_fold_data_file_no_clf = os.path.join(
            results_dir,
            fname + "_without_classifier" + fext,
        )

        # ------------------------------
        # Training
        # ------------------------------

        print("== Training on all folds except Fold {}".format(i_split))

        if os.path.isfile(i_fold_data_file) and (not force_training):
            print("   - Skipping training, fold training data already exists")

            # Load fold data if it exists and were are not forcing the training
            if os.path.isfile(i_fold_data_file_no_clf):
                i_fold_data = uf.load_model_from_pickle(i_fold_data_file_no_clf)
            else:
                i_fold_data = uf.load_model_from_pickle(i_fold_data_file)
        else:
            # ------------------------------
            # Step 0: Read and preprocess training data
            # ------------------------------

            print("   - Loading training data")
            (
                train_x,
                train_y,
                train_missing_labels,
                train_timestamps,
                feature_names,
                all_labels,
            ) = uf.read_fold_data(i_train_uuids, data_dir)

            feature_sensor_names = uf.get_sensor_names_from_features(feature_names)

            train_x, train_y = clean_data(train_x, train_y, all_labels, activities_list)

            # ------------------------------
            # Step 1: Randomly sample X% of the features of each sensor
            # ------------------------------
            rng = np.random.default_rng(config[fusion_method]["random_seed"] + i_split)
            features_ratio = config[fusion_method]["percentage_of_features"] / 100.0
            num_reduced_sets = config[fusion_method]["reduced_feature_sets"]
            reduced_set_indices = find_reduced_set_indices(
                num_reduced_sets,
                features_ratio,
                sensor_indices,
                sensors_list,
                rng,
            )

            # ------------------------------
            # Step 2: Find samples to be used for each augmented class combination
            # ------------------------------

            # Mask of positive (and negative) samples in proposition
            # combination
            combo_positive_samples = []

            for j, j_pos_combo in enumerate(positive_augmented_class_idx):
                positive_class_mask = np.any(train_y[:, j_pos_combo], axis=1)
                num_positive_samples = positive_class_mask.sum()

                combo_positive_samples.append(
                    {
                        "positive_class_indices": j_pos_combo,
                        "negative_class_indices": negative_augmented_class_idx[j],
                        "positive_class_mask": positive_class_mask,
                        "negative_class_mask": np.logical_not(positive_class_mask),
                        "num_positive_samples": num_positive_samples,
                        "num_negative_samples": positive_class_mask.size
                        - num_positive_samples,
                    }
                )

            # ------------------------------
            # Step 3: Train using SMOTE and bagging
            # ------------------------------

            # Each fold takes about 6 min to train 5 reduced sets on current config
            with Timer("Model training duration"):
                print("   - Training classifiers")
                i_reduced_set_models = train_model(
                    train_x,
                    train_y,
                    train_missing_labels,
                    feature_sensor_names,
                    all_labels,
                    sensors_list,
                    activities_list,
                    reduced_set_indices,
                    combo_positive_samples,
                    config[fusion_method],
                )

            i_fold_data = {
                "reduced_set_models": i_reduced_set_models,
                "positive_augmented_class_idx": positive_augmented_class_idx,
                "reduced_set_indices": reduced_set_indices,
                "combo_positive_samples": combo_positive_samples,
            }
            uf.save_model_to_pickle(
                i_fold_data_file,
                i_fold_data,
            )



        # ------------------------------
        # Step 4: Test
        # ------------------------------

        print("== Testing on Fold {}".format(i_split))

        if os.path.isfile(
            os.path.join(results_dir, f"fold_{num_splits-1}_predictions.pkl")
        ) and (not force_test):
            print("[CODE NOT IMPLEMENTED]")
            sys.exit(1)
        else:
            # ------------------------------
            # Load and preprocess test data
            # ------------------------------
            print("   - Loading test data")
            (
                test_x,
                test_y,
                test_missing_labels,
                test_timestamps,
                feature_names,
                all_labels,
            ) = uf.read_fold_data(i_test_uuids, data_dir)

            # Remove samples/observation with missing labels in test data
            test_x, test_y = clean_data(test_x, test_y, all_labels, activities_list)
            num_test_samples = len(test_x)

            # ------------------------------
            # Run all P x K models on the test samples/observations
            # ------------------------------

            num_reduced_sets = len(i_fold_data["reduced_set_models"])
            num_class_combos = len(i_fold_data["reduced_set_models"][0])

            predictions_before_fusion_file = os.path.join(
                results_dir,
                "fold_{}_predictions_before_fusion.pkl".format(i_split),
            )

            # Save the before fusion results to avoid unnecessary recalculation
            if not os.path.isfile(predictions_before_fusion_file):
                # First dim is class proposition combination; second dim is reduced
                # feature set
                predictions_before_fusion = []
                for k in range(num_class_combos):
                    # print(f"k = {k}")
                    predictions_before_fusion.append(
                        [
                            i_fold_data["reduced_set_models"][j][k][
                                "bagging_classifier"
                            ].predict_proba(
                                test_x[
                                    :,
                                    i_fold_data["reduced_set_indices"][j][
                                        "sampled_features"
                                    ],
                                ]
                            )
                            for j in range(num_reduced_sets)
                        ]
                    )

                uf.save_model_to_pickle(
                    predictions_before_fusion_file, predictions_before_fusion
                )
            else:
                print("   - Loading predictions before fusion")
                predictions_before_fusion = uf.load_model_from_pickle(
                    predictions_before_fusion_file
                )

            # Sanity check: What is the observed performance of all P x K models?
            # TODO: Save results per split
            for k in range(num_class_combos):
                pos_class_idx = i_fold_data["reduced_set_models"][0][k][
                    "positive_class_indices"
                ]
                pos_class_labels = i_fold_data["reduced_set_models"][0][k][
                    "positive_class_activities"
                ]
                neg_class_idx = i_fold_data["reduced_set_models"][0][k][
                    "negative_class_indices"
                ]
                neg_class_labels = i_fold_data["reduced_set_models"][0][k][
                    "negative_class_activities"
                ]
                dir_name = os.path.join(
                    results_dir,
                    "binary_model_results",
                    "{}__vs__{}".format(
                        "__".join(pos_class_labels).lower(),
                        "__".join(neg_class_labels).lower(),
                    ),
                )

                # Create folder to store results per class combination
                Path(dir_name).mkdir(parents=True, exist_ok=True)

                # NOTE: 1 stands for positive class
                for j in range(num_reduced_sets):
                    binary_pred_y = np.argmax(predictions_before_fusion[k][j], axis=1)

                    binary_true_y = np.zeros(binary_pred_y.shape, dtype=int)
                    z = test_y[:, pos_class_idx]
                    z = z.sum(axis=1)
                    binary_true_y[z == 1] = 1

                    binary_perf_metrics = uf.perf_measure(binary_true_y, binary_pred_y)

                    with open(
                        os.path.join(dir_name, f"reduced_set_{j:02}.csv"), "w"
                    ) as f:
                        w = csv.DictWriter(f, binary_perf_metrics.keys())
                        w.writeheader()
                        w.writerow(binary_perf_metrics)

            # TODO: This must be done after predictions_before_fusion is computed
            # Save fold data without bagging classifier
            for i_set in i_fold_data["reduced_set_models"]:
                for j_combo in i_set:
                    del j_combo["bagging_classifier"]
            fname, fext = os.path.splitext(i_fold_data_file)
            uf.save_model_to_pickle(
                fname + "_without_classifier" + fext,
                i_fold_data,
            )

            # Skip testing if needed
            if config[fusion_method]["skip_testing"]:
                continue

            # ------------------------------
            # Other important stuff
            # ------------------------------

            naps_models = []
            for k in range(num_class_combos):
                naps_models.append(
                    [
                        copy.deepcopy(
                            i_fold_data["reduced_set_models"][j][k]["ds_model"]
                        )
                        for j in range(num_reduced_sets)
                    ]
                )

            # Calculate the number of top performing models to use in fusion
            num_top_models = config[fusion_method]["number_top_models"]
            if type(num_top_models) == float:
                num_top_models = int(np.round(num_top_models * num_reduced_sets))
            else:
                assert type(num_top_models) == int

            num_chosen_indices = config[fusion_method]["test_ratio"]
            if type(num_chosen_indices) == float:
                num_chosen_indices = int(
                    np.round(num_test_samples * num_chosen_indices)
                )
            else:
                assert (type(num_chosen_indices) == int) and (
                    num_chosen_indices <= num_test_samples
                )

            rng = np.random.default_rng(config[fusion_method]["random_seed"] + i_split)
            chosen_idx = rng.choice(
                num_test_samples, size=num_chosen_indices, replace=False
            )
            chosen_idx.sort()
            num_chosen_idx = len(chosen_idx)
            i_fusion_pred_labels_mat = np.zeros(
                (num_chosen_idx, num_activities), dtype=int
            )

            custom_fmt = "%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d remaining test samples"
            progress = naps.ProgressBar(num_chosen_idx, fmt=custom_fmt)
            for n, i in enumerate(chosen_idx):
                # NOTE: Find out why we initialize to max uncertainty
                uncertainty_mat = np.ones([num_reduced_sets, num_class_combos])

                for j in range(num_reduced_sets):
                    for k in range(num_class_combos):
                        pred_prob = predictions_before_fusion[k][j][i]

                        # Calculate mean of class imbalance uncertainty
                        # (Uncertainty_B) and voting uncertainty
                        # (Uncertainty_Context)
                        uncertainty_mat[j, k] = (
                            naps_models[k][j].Uncertainty_B
                            + naps_models[k][j].Uncertainty_Context(pred_prob)
                        ) / 2

                        naps_models[k][j].Mass_Function_Setter(
                            uncertainty_mat[j, k], pred_prob
                        )

                # Top model selection
                # NOTE: Specifying 0 for fx_axis in naps.Model_Selector() doesn't really work
                index_m = np.argsort(uncertainty_mat, axis=0).T
                selected_models_idx = index_m[:, 0:num_top_models]

                pred_y = naps.Fuse_and_Predict(
                    selected_models_idx,
                    naps_models,
                    activities_list,
                    num_activities,
                    num_class_combos,
                    num_top_models,
                )

                i_fusion_pred_labels_mat[n] = pred_y

                progress.current += 1
                progress()

            progress.done()

            # ------------------------------
            # Save results
            # ------------------------------

            i_fusion_true_labels_mat = test_y[chosen_idx]

            # Find label indices
            i_fusion_true_labels = np.argmax(i_fusion_true_labels_mat, axis=1)
            i_fusion_pred_labels = np.argmax(i_fusion_pred_labels_mat, axis=1)

            # Save prediction results
            i_fusion_results = {
                "y": i_fusion_true_labels,
                "y_pred": i_fusion_pred_labels,
            }
            uf.perf_save(
                i_fusion_results,
                results_dir,
                i_split,
            )

            all_true_y = i_fusion_true_labels
            all_predicted_y = i_fusion_pred_labels

            # Calculate performance metric using PyCM
            perf_metrics_pycm = ConfusionMatrix(
                actual_vector=all_true_y, predict_vector=all_predicted_y
            )
            label_pretty_names = {
                i: uf.get_label_pretty_name(label.lower())
                for (i, label) in enumerate(activities_list)
            }
            perf_metrics_pycm.relabel(mapping=label_pretty_names)
            print(perf_metrics_pycm)

            # Calculate performance metrics (stored in dict) using code found online.
            # This shows balanced accuracy, which PyCM does not calculated.
            conf_mat = confusion_matrix(all_true_y, all_predicted_y)
            perf_metrics = nmc.get_performance_metrics(conf_mat)
            print("Balanced accuracy =", perf_metrics["balanced_accuracy"])

            nmc.plot_confusion_matrix(
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


def main():
    with open(os.path.join("../config.toml"), "rb") as f:
        project_config = tomllib.load(f)

    # fusion_method = project_config["fusion_method"]
    #
    # data_dir = project_config["data_dir"]
    # folds_dir = project_config["cross_validation_dir"]
    # results_dir = project_config[fusion_method]["results_dir"]
    # sensors_list = project_config[fusion_method]["sensor_combination"]
    # target_labels = project_config[fusion_method]["activities"]

    # Generate models and results
    run_cross_val(project_config)

    # summarize_performance(
    #     os.path.join(results_dir, "results_" + fusion_method),
    #     target_labels,
    # )


if __name__ == "__main__":
    with Timer("main()"):
        main()
