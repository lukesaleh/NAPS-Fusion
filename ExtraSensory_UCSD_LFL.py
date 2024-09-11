# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:21:46 2018

@author: Mehrdad
"""
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import glob
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def perf_measure(y_actual, y_hat):
    """This function gets the test set response and predicted response and
    returns the "true positive", "true negative", "false positive" and "false
    negative" """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return (TP, FP, TN, FN)


def get_folds_uuids(fold_dir):
    """
    The function gets the directory where the folds text files are located
    and returns a list of five np.arrays. In each array, the uuids of the
    corresponding fold are stored.
    """
    # folds_uuids is a list of np.arrays. Each array is a set of uuids
    folds_uuids = [0, 1, 2, 3, 4]
    num_folds = len(folds_uuids)

    # This loop reads all 5 test folds (iphone and android) and stores uuids
    for i in range(0, num_folds):
        filename = "fold_{}_test_android_uuids.txt".format(i)
        filepath = os.path.join(fold_dir, filename)
        # aux1 is the uuids of ith test fold for "android"
        aux1 = pd.read_csv(filepath, header=None)
        aux1 = aux1.values

        filename = "fold_%s_test_iphone_uuids.txt" % i
        filepath = os.path.join(fold_dir, filename)
        # aux2 is the uuids of ith test fold for "iphone"
        aux2 = pd.read_csv(filepath, header=None)
        aux2 = aux2.values

        # Then we concatenate them
        folds_uuids[i] = np.concatenate((aux1, aux2), axis=0)

    return folds_uuids


# TODO: Remove, not used
def get_filepath(file_dir, uuid):
    """This function gets the uuid of a subject and returns the file path for
    csv file of the subject"""
    filename = "{}.features_labels.csv".format(uuid)
    filepath = os.path.join(file_dir, filename)
    return filepath


def readdata_csv(data_dir):
    """This function gets the directory of the datasets and returns the dataset
    containing information of all 60 users"""
    length_uuids = 36  # number of characters for each uuid
    data_list = glob.glob(os.path.join(os.getcwd(), data_dir, "*.csv"))
    # grand_dataset is a dict that holds the uuids and corresponding dataset
    grand_dataset = {}
    length_of_dataset = 0
    for i in range(len(data_list)):
        # dismantles the file name and picks only uuids (first 36 characters)
        uuid = os.path.basename(data_list[i])[:length_uuids]
        dataset_ith = pd.read_csv(data_list[i])
        print("Dimensions of dataset {}:".format(i), dataset_ith.shape)
        length_of_dataset += len(dataset_ith)
        grand_dataset[uuid] = dataset_ith
    print("Grand total of examples: {}".format(length_of_dataset))
    return grand_dataset


def train_test_split(test_fold, num_folds, fold_dir, grand_dataset):
    """This function takes the number of test fold (ranging from 0 to 4) and
    number of folds (in this case 5) and directory where the folds' uuids are
    and the dataset, and returns train and test datasets"""
    train_dataset = pd.DataFrame()
    test_dataset = pd.DataFrame()
    folds_uuids = get_folds_uuids(fold_dir)

    # Dividing the folds uuids into train and test (the L denotes they are still lists)
    test_uuids_L = [folds_uuids[test_fold]]

    # Remove the test fold from the list to come up with train UUIDs
    del folds_uuids[test_fold]
    train_uuids_L = folds_uuids

    # Transforming the list of arrays of uuids into a single uuids np.array
    test_uuids = np.vstack(test_uuids_L)
    train_uuids = np.vstack(train_uuids_L)

    # Now collecting the test and train dataset using concatenating
    for i in train_uuids:
        train_dataset = pd.concat([train_dataset, grand_dataset[i[0]]])

    for j in test_uuids:
        test_dataset = pd.concat([test_dataset, grand_dataset[j[0]]])

    return (train_dataset, test_dataset)


def pred_resp(
    train_dataset, test_dataset, target_sensor, target_activity, target_sensors_fusion
):
    """This function takes the train and test datasets and the target sensor
    and the target activity to model, then returns four dataframes: X_train,
    X_test, y_train and y_test (It also remove the rows with nan value and
    normalizes the features X"""
    activity, Sensor = labeling()
    # "feat_train" and "resp_train" are selected based off of target sensor and activity
    feat_train = train_dataset.iloc[:, Sensor[target_sensor]]
    # feat_train[np.isnan(feat_train)] = 0.
    # feat_train = feat_train.fillna(0)
    resp_train = train_dataset.iloc[:, activity[target_activity]]
    # Then we concatenate them and eliminate rows with "nan" values
    feat_resp_train = pd.concat([feat_train, resp_train], axis=1)
    feat_resp_train = feat_resp_train.dropna()
    # We again split them into X and y (feature matrix and response array)
    X_train = feat_resp_train.iloc[:, 0:-1]
    # We also standardize (or normalize) the features
    X_train = preprocessing.scale(X_train)
    y_train = feat_resp_train.iloc[:, -1]

    # Same goes with test dataset
    feat_test = pd.DataFrame()
    resp_test = pd.DataFrame()
    feat_test = test_dataset.iloc[:, Sensor[target_sensor]]
    # feat_test[np.isnan(feat_test)] = 0.
    # feat_test = feat_test.fillna(0)
    resp_test = test_dataset.iloc[:, activity[target_activity]]
    feat_resp_test = pd.concat([feat_test, resp_test], axis=1)
    feat_resp_test = feat_resp_test.dropna()
    X_test = feat_resp_test.iloc[:, 0:-1]
    X_test = preprocessing.scale(X_test)
    y_test = feat_resp_test.iloc[:, -1]

    feat_test = pd.DataFrame()
    resp_test = pd.DataFrame()
    for i in range(len(target_sensors_fusion)):
        feat_test = pd.concat(
            [feat_test, test_dataset.iloc[:, Sensor[target_sensors_fusion[i]]]], axis=1
        )
    resp_test = test_dataset.iloc[:, activity[target_activity]]
    feat_resp_test = pd.concat([feat_test, resp_test], axis=1)
    feat_resp_test = feat_resp_test.dropna()

    t_s_f_l = []
    for i in range(len(target_sensors_fusion)):
        t_s_f_l.append(len(Sensor[target_sensors_fusion[i]]))

    ind_target_sensor = target_sensors_fusion.index(target_sensor)

    start = 0

    for i in range(0, ind_target_sensor):
        start += t_s_f_l[i]
    X_train_f = feat_resp_train.iloc[:, start : (start + t_s_f_l[ind_target_sensor])]
    y_train_f = feat_resp_train.iloc[:, -1]

    X_test_f = feat_resp_test.iloc[:, start : (start + t_s_f_l[ind_target_sensor])]
    y_test_f = feat_resp_test.iloc[:, -1]

    return (X_train, y_train, X_test, y_test, X_train_f, y_train_f, X_test_f, y_test_f)


def LFL(y_prob):
    """This function gets the predicted probabilities of all sensors for one
    specific activity and average them and returns a binary y_pred that can be
    easily compared with y_test"""
    # TODO: This seems fishy
    y_pred = np.floor(y_prob)
    return y_pred


def labeling():
    """This function defines two dictionaries for activities and sensors. Each
    dictionary holds the range of columns for the specified sensor or
    activity"""
    Lying_down = 226
    activity = {}
    #    activity['Lying_down'] = Lying_down
    activity["Sitting"] = Lying_down + 1
    #    activity['Walking'] = Lying_down + 2
    #    activity['Running'] = Lying_down + 3
    #    activity['Bicylcing'] = Lying_down + 4
    # activity['Standing'] = Lying_down + 44
    #    activity['Sleeping'] = Lying_down + 5
    #    activity['Lab_work'] = Lying_down + 6
    #    activity['In_class'] = Lying_down + 7
    #    activity['In_meeting'] = Lying_down + 8
    #    activity['At_workplace'] = Lying_down + 9
    #    activity['Indoors'] = Lying_down + 10
    #    activity['Outside'] = Lying_down + 11
    #    activity['In_car'] = Lying_down + 12
    #    activity['On_bus'] = Lying_down + 13
    #    activity['Driver'] = Lying_down + 14
    #    activity['Passenger'] = Lying_down + 15
    #    activity['At_home'] = Lying_down + 16
    #    activity['At_restaurant'] = Lying_down + 17
    #    activity['Phone_in_pocket'] = Lying_down + 18
    #    activity['Excercise'] = Lying_down + 19
    #    activity['Cooking'] = Lying_down + 20
    #    activity['Shopping'] = Lying_down + 21
    #    activity['Strolling'] = Lying_down + 22
    #    activity['Drinking'] = Lying_down + 23
    #    activity['Bathing'] = Lying_down + 24

    Sensor = {}
    Sensor["Acc"] = range(1, 27)
    Sensor["Gyro"] = range(27, 53)
    Sensor["W_acc"] = range(84, 130)
    Sensor["Loc"] = range(139, 156)
    Sensor["Aud"] = range(156, 182)
    Sensor["PS"] = np.append(range(184, 210), range(218, 226))

    return (activity, Sensor)


#########################################################
# ---------------- Main program starts ------------------#
# ---------------- Single Sensor Classifier (clf) -------#
#########################################################
if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        project_config = tomllib.load(f)

    datadir = project_config["data_dir"]
    cvdir = project_config["cross_validation_dir"]

    num_folds = 5

    # Select the activity and sensors to use
    activity, Sensor = labeling()  # creating two dicts for sensor and activity

    dataset = readdata_csv(datadir)  # reading all data and storing in "dataset" a DF

    ##################################################################
    # ----------------------- Sensor Fusion --------------------------#
    ##################################################################

    # Defining dictionary for "True Positives", "True Negatives" and etc.
    # Also defining performance metrics
    TP = {}
    TN = {}
    FP = {}
    FN = {}
    recall = {}
    specifity = {}  # specificity
    accuracy = {}
    precision = {}
    blncd_aqrc = {}  # balanced accuracy
    f1 = {}

    # Defining the same variables for Late Fusion Learned Weights (LFL)
    TP_LFL = {}
    TN_LFL = {}
    FP_LFL = {}
    FN_LFL = {}
    recall_LFL = {}
    specifity_LFL = {}
    accuracy_LFL = {}
    precision_LFL = {}
    blncd_aqrc_LFL = {}
    f1_LFL = {}

    for a in activity:
        TP[a] = {}
        TN[a] = {}
        FP[a] = {}
        FN[a] = {}
        recall[a] = {}
        specifity[a] = {}
        accuracy[a] = {}
        precision[a] = {}
        blncd_aqrc[a] = {}
        f1[a] = {}

        TP_LFL[a] = 0
        TN_LFL[a] = 0
        FP_LFL[a] = 0
        FN_LFL[a] = 0
        recall_LFL[a] = 0
        specifity_LFL[a] = 0
        accuracy_LFL[a] = 0
        precision_LFL[a] = 0
        blncd_aqrc_LFL[a] = 0
        f1_LFL[a] = 0

        for s in Sensor:
            TP[a][s] = 0
            FP[a][s] = 0
            TN[a][s] = 0
            FN[a][s] = 0
            recall[a][s] = 0
            specifity[a][s] = 0
            accuracy[a][s] = 0
            precision[a][s] = 0
            blncd_aqrc[a][s] = 0
            f1[a][s] = 0

    ##################################################################
    # ----------------------- ML Algorithm ---------------------------#
    ##################################################################

    all_sensors_f = list(Sensor)

    for a in activity:  # looping over all activities
        print(a)
        #        i==num_folds[0]
        for i in range(num_folds):
            print("CV=", i)
            # Looping over number of folds

            m = 0  # this is for tracking the sensors
            # Getting the test and train dataset
            # if i==0:

            train_dataset, test_dataset = train_test_split(i, num_folds, cvdir, dataset)
            #
            for s in Sensor:
                print(s)
                (
                    X_train,
                    y_train,
                    X_train_f,
                    y_train_f,
                    X_test,
                    y_test,
                    X_test_f,
                    y_test_f,
                ) = pred_resp(train_dataset, test_dataset, s, a, all_sensors_f)
                if m == 0:
                    y_prob = np.zeros([len(y_test_f), len(all_sensors_f)])
                    print("Getting Probability values")
                #                clf = LogisticRegression(class_weight='balanced' ,C=0.01)
                # TODO: Investigate the use of multi_class="multinomial" and
                #  C=0.001 (inverse of regularization strength). See
                #  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
                clf = LogisticRegression(
                    multi_class="multinomial", solver="lbfgs", C=0.001
                )
                # clf = LogisticRegression(multi_class='ovr', solver='sag',  C=1.0, penalty ='l2')
                # clf = LogisticRegression( random_state=0, class_weight='balanced', solver='newton-cg')  # Walking
                # clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg', C=1.0)
                # clf = LogisticRegression(random_state=0, class_weight='balanced', solver='lbfgs')
                # clf = LogisticRegression(random_state=0, class_weight='balanced', multi_class='multinomial', C=1.0, solver='lbfgs')
                #                param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
                #                clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
                #                GridSearchCV(cv=None, estimator=LogisticRegression(C=1.0, class_weight='balanced', intercept_scaling=1, dual=False, fit_intercept=True, penalty='l2', tol=0.0001), param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
                clf = clf.fit(X_train_f, y_train_f)

                y_prob[:, m] = clf.predict_proba(X_test_f)[:, 1]
                yprob_df = pd.DataFrame.from_dict(y_prob)
                yprobdf_nozeros = yprob_df.loc[:, (yprob_df != 0).any(axis=0)]
                target_path = os.path.join(
                    project_config["lfl-fusion"]["results_dir"], "y_prob", str(i) + s
                )
                filename = target_path + "_yprob.csv"
                yprob_ss = yprobdf_nozeros.to_csv(
                    filename, sep=",", encoding="utf-8", index=False, header=s
                )

            interesting_files = glob.glob(
                os.path.join(
                    project_config["lfl-fusion"]["results_dir"], "y_prob", "*.csv"
                )
            )
            df_list = []
            for filename in sorted(interesting_files):
                print("Processing", filename)
                df_list.append(pd.read_csv(filename))
                full_df = pd.concat(df_list, axis=1)
                # save the final file in same directory:
                full_df.to_csv(
                    os.path.join(
                        project_config["lfl-fusion"]["results_dir"],
                        "y_prob",
                        "merged_pandas.csv",
                    ),
                    index=True,
                    encoding="utf-8-sig",
                    header=s,
                )
                X_train_new_df = pd.DataFrame(full_df)

            X_train_new_df = pd.DataFrame(full_df)
            clf_2 = LogisticRegression(
                class_weight="balanced", C=0.01
            )  # # Typically, the data is highly imbalanced, with many more negative examples;
            # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
            # clf_2 = LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs', C=100.0) #(random_state=0, multi_class='multinomial', solver='lbfgs',
            # TODO: This is where I stopped following. This is also where code fails
            model = clf_2.fit(X_train_new_df, y_test_f)
            clf_2.coef_.shape
            clf_2.intercept_.shape
            print("Done with second classifier")
            # y_test_f = np.array(y_test_f)
            #                new_observation = [[.5,.5,.5,.5,.5,.5]]
            #                new_observation = y_test_f,axis=0 #,axis=1, [0].reshape(-1,1)
            y_pred_new = model.predict(X_train_new_df)
            # accuracy_score(y_test_f, y_pred_new)
            y_prob_new = model.predict_proba(X_train_new_df)

            #
            #            # Applyingthe late fusion average
            y_pred_LFL = LFL(y_prob_new)
            y_test_f = np.array(y_test_f)
            TP_temp, FP_temp, TN_temp, FN_temp = perf_measure(y_test_f, y_pred_new)

            TP_LFL[a] += TP_temp
            FP_LFL[a] += FP_temp
            TN_LFL[a] += TN_temp
            FN_LFL[a] += FN_temp
    #
    #                TP_temp, FP_temp, TN_temp, FN_temp = perf_measure(y_test,y_pred_new)
    #
    #                TP[a][s] += TP_temp
    #                FP[a][s] += FP_temp
    #                TN[a][s] += TN_temp
    #                FN[a][s] += FN_temp
    #
    #                m = m + 1
    #
    #            # Applyingthe late fusion average
    #            y_pred_LFL = LFL(y_prob_new)
    #            y_test_f = np.array(y_test_f)
    #            TP_temp, FP_temp, TN_temp, FN_temp = perf_measure(y_test_f,y_pred_LFL)
    #
    #            TP_LFL[a] += TP_temp
    #            FP_LFL[a] += FP_temp
    #            TN_LFL[a] += TN_temp
    #            FN_LFL[a] += FN_temp

    ####################################################
    # ------------- performance measurement ------------#
    ####################################################

    # Knowing the TP, TN, FP, FN for each pair of activity-sensor, it is
    # a straightforward task to compute the metrics...
    for a in activity:
        accuracy_LFL[a] = (TP_LFL[a] + TN_LFL[a]) / (
            TP_LFL[a] + TN_LFL[a] + FP_LFL[a] + FN_LFL[a]
        )
        recall_LFL[a] = TP_LFL[a] / (TP_LFL[a] + FN_LFL[a])
        specifity_LFL[a] = TN_LFL[a] / (TN_LFL[a] + FP_LFL[a])
        precision_LFL[a] = TP_LFL[a] / (TP_LFL[a] + FP_LFL[a])
        blncd_aqrc_LFL[a] = (recall_LFL[a] + specifity_LFL[a]) / 2
        f1_LFL[a] = (2 * recall_LFL[a] * precision_LFL[a]) / (
            recall_LFL[a] + precision_LFL[a]
        )

    # Transforming dicts to dataframes
    f1_ss = pd.DataFrame.from_dict(f1)
    f1_ss = f1_ss.transpose()
    f1_LFL = pd.Series(f1_LFL)
    f1_LFL = pd.DataFrame(f1_LFL, columns=["LFL"])
    f1 = pd.concat([f1_ss, f1_LFL], axis=1)
    # Saving the dataframe to csv file
    f1.to_csv(
        os.path.join(
            project_config["lfl-fusion"]["results_dir"], "metrics", "f1_score.csv"
        )
    )

    recall_ss = pd.DataFrame.from_dict(recall)
    recall_ss = recall_ss.transpose()
    recall_LFL = pd.Series(recall_LFL)
    recall_LFL = pd.DataFrame(recall_LFL, columns=["LFL"])
    recall = pd.concat([recall_ss, recall_LFL], axis=1)
    recall.to_csv(
        os.path.join(
            project_config["lfl-fusion"]["results_dir"], "metrics", "recall.csv"
        )
    )

    precision_ss = pd.DataFrame.from_dict(precision)
    precision_ss = precision_ss.transpose()
    precision_LFL = pd.Series(precision_LFL)
    precision_LFL = pd.DataFrame(precision_LFL, columns=["LFL"])
    precision = pd.concat([precision_ss, precision_LFL], axis=1)
    precision.to_csv(
        os.path.join(
            project_config["lfl-fusion"]["results_dir"], "metrics", "precision.csv"
        )
    )

    accuracy_ss = pd.DataFrame.from_dict(accuracy)
    accuracy_ss = accuracy_ss.transpose()
    accuracy_LFL = pd.Series(accuracy_LFL)
    accuracy_LFL = pd.DataFrame(accuracy_LFL, columns=["LFL"])
    accuracy = pd.concat([accuracy_ss, accuracy_LFL], axis=1)
    accuracy.to_csv(
        os.path.join(
            project_config["lfl-fusion"]["results_dir"], "metrics", "accuracy.csv"
        )
    )

    blncd_aqrc_ss = pd.DataFrame.from_dict(blncd_aqrc)
    blncd_aqrc_ss = blncd_aqrc_ss.transpose()
    blncd_aqrc_LFL = pd.Series(blncd_aqrc_LFL)
    blncd_aqrc_LFL = pd.DataFrame(blncd_aqrc_LFL, columns=["LFL"])
    blncd_aqrc = pd.concat([blncd_aqrc_ss, blncd_aqrc_LFL], axis=1)
    blncd_aqrc.to_csv(
        os.path.join(
            project_config["lfl-fusion"]["results_dir"], "metrics", "blncd_aqrc.csv"
        )
    )

    specifity_ss = pd.DataFrame.from_dict(specifity)
    specifity_ss = specifity_ss.transpose()
    specifity_LFL = pd.Series(specifity_LFL)
    specifity_LFL = pd.DataFrame(specifity_LFL, columns=["LFL"])
    specifity = pd.concat([specifity_ss, specifity_LFL], axis=1)
    specifity.to_csv(
        os.path.join(
            project_config["lfl-fusion"]["results_dir"], "metrics", "specifity.csv"
        )
    )
