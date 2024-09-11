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
    The function gets the directory where the the folds text files are located
    and returns a list of five np.arrays in each of them the uuids of the
    corresponding fold are stored.
    """
    num_folds = 5
    # folds_uuids is gonna be a list of np.arrays. each array is a set of uuids
    folds_uuids = [0, 1, 2, 3, 4]
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
    # grand_dataset is a dict. that holds the uuids and correspondong datast
    grand_dataset = {}
    lengthOFdataset = 0
    for i in range(len(data_list)):
        #    for i in range(5):
        # dismantles the file name and picks only uuids (first 36 characters)
        uuid = os.path.basename(data_list[i])[:length_uuids]
        dataset_ith = pd.read_csv(data_list[i])
        print(i, dataset_ith.shape)
        lengthOFdataset += len(dataset_ith)
        grand_dataset[uuid] = dataset_ith
    print(lengthOFdataset)
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


# def pred_resp(train_dataset,test_dataset,target_sensor,target_activity):
#    """This function takes the train and test datasets and the target sensor
#    and the target activity to model, then returns four dataframes: X_train,
#    X_test, y_train and y_test (It also remove the rows with nan value and
#    normalizes the features X"""
#    activity,Sensor = labeling()
#    # "feat_train" and "resp_train" are selected based off of target sensor and activity
#    feat_train = train_dataset.iloc[:, Sensor[target_sensor]]
#    resp_train = train_dataset.iloc[:,activity[target_activity]]
#    # Then we concatenate them and eliminate rows with "nan" values
#    feat_resp_train = pd.concat([feat_train,resp_train],axis=1)
#    feat_resp_train = feat_resp_train.dropna()
#    # We again split them into X and y (feature matrix and response array)
#    X_train = feat_resp_train.iloc[:,0:-2]
#    # We also standardize (or normalize) the features
#    X_train = preprocessing.scale(X_train)
#    y_train = feat_resp_train.iloc[:,-1]
#
#     # Same goes with test dataset
#    feat_test = test_dataset.iloc[:,Sensor[target_sensor]]
#    resp_test = test_dataset.iloc[:,activity[target_activity]]
#    feat_resp_test = pd.concat([feat_test,resp_test],axis=1)
#    feat_resp_test = feat_resp_test.dropna()
#    X_test = feat_resp_test.iloc[:,0:-2]
#    X_test = preprocessing.scale(X_test)
#    y_test = feat_resp_test.iloc[:,-1]


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

    X_test_f = feat_resp_test.iloc[:, start : (start + t_s_f_l[ind_target_sensor])]
    X_test_f = preprocessing.scale(X_test_f)
    y_test_f = feat_resp_test.iloc[:, -1]

    return (X_train, y_train, X_test, y_test, X_test_f, y_test_f)


def LFA(y_prob):
    """This funcrion gets the predicted probabilities of all sensors for one
    specific activity and average them and returns a binary y_pred that can be
    easily compared with y_test"""
    y_pred = np.floor(np.mean(y_prob, axis=1) * 2)
    return y_pred


def labeling():
    """This function defines two dictionaries for activities and sensors. Each
    dictionaray holds the the range of columns for the specified sensor or
    activity"""
    Lying_down = 226
    activity = {}
    activity["Lying_down"] = Lying_down
    activity["Sitting"] = Lying_down + 1
    activity["Walking"] = Lying_down + 2
    activity["Running"] = Lying_down + 3
    activity["Bicylcing"] = Lying_down + 4
    activity["Sleeping"] = Lying_down + 5
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
    #    Sensor['Mag'] = range(53,84)
    Sensor["W_acc"] = range(84, 130)
    #    Sensor['Compass'] = range(130,139)
    Sensor["Loc"] = range(139, 156)
    Sensor["Aud"] = range(156, 182)
    #    Sensor['AP'] = range(182,184)
    #    Sensor['PS'] = np.append(range(184,210),range(218,226))
    #    Sensor['LF'] = range(210,218)

    return (activity, Sensor)


#########################################################
# ---------------- Main program starts ------------------#
# ---------------- Single Sensor clf --------------------#
#########################################################
if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        project_config = tomllib.load(f)

    datadir = project_config["data_dir"]
    cvdir = project_config["cross_validation_dir"]

    num_folds = 5

    activity, Sensor = labeling()  # creating two dicts for sensor and activity
    dataset = readdata_csv(datadir)  # reading all data and storing in "dataset" a DF

    # Now we create dict. of dict. for varialbes required to measure accuracy
    # In each dict. we have keys that are name of the "sensors" and for each
    # key we have another dict. of name of activity and corresponding values.
    # we will transform this to a pandas dataframe later on
    #    TP = {}
    #    TN = {}
    #    FP = {}
    #    FN = {}
    #    recall = {}
    #    specifity = {}
    #    accuracy = {}
    #    precision = {}
    #    blncd_aqrc = {}
    #    f1 = {}
    #
    #    TP_LFA = {}
    #    TN_LFA = {}
    #    FP_LFA = {}
    #    FN_LFA = {}
    #    recall_LFA = {}
    #    specifity_LFA = {}
    #    accuracy_LFA = {}
    #    precision_LFA = {}
    #    blncd_aqrc_LFA = {}
    #    f1_LFA = {}

    # Two for loops for "activity" and "sensor" to build a classifier for every
    # pair of "sensor-activity"
    #    for s in Sensor:
    #        ss = 0
    #        print(s)
    #        TP[s] = {}
    #        TN[s] = {}
    #        FP[s] = {}
    #        FN[s] = {}
    #        recall[s] = {}
    #        specifity[s] = {}
    #        accuracy[s] = {}
    #        precision[s] = {}
    #        blncd_aqrc[s] = {}
    #        f1[s] = {}
    #        clf[s] = {}
    #
    #        for a in activity:
    #            print(a)
    #            # Initializing values of the dicts to 0
    #            TP[s][a] = 0
    #            FP[s][a] = 0
    #            TN[s][a] = 0
    #            FN[s][a] = 0
    #            recall[s][a] = 0
    #            specifity[s][a] = 0
    #            accuracy[s][a] = 0
    #            precision[s][a] = 0
    #            blncd_aqrc[s][a] = 0
    #            f1[s][a] = 0
    #            clf[s][a] = 0
    #
    #            # This for loop does the cross-validation. In each iteration one fold
    #            # is specified as the test set and the rest are train set
    #            for i in range(num_folds):
    #                # Getting the test and train dataset
    #                train_dataset,test_dataset = \
    #                train_test_split(i,num_folds,cvdir,dataset)
    #
    #                # Getting X-train, X_test, y_train, y_test
    #                X_train,y_train,X_test,y_test = \
    #                pred_resp(train_dataset,test_dataset,s,a)
    #
    #                # Removing train and test dataset to free up the space
    #                del(train_dataset,test_dataset)
    #
    #                # Building a logistic regression classifier. Class weights are set
    #                # to 'balanced' to take care of unbalanced labels (like in vizman paper)
    #                clf[s][a] = LogisticRegression(random_state=0, class_weight='balanced',\
    #                      solver='lbfgs').fit(X_train, y_train)
    #
    #                y_pred_prob = clf[s][a].predict_proba(X_test)
    #                y_pred = clf[s][a].predict(X_test)
    #                y_test = y_test.as_matrix(columns=None)
    #
    #                # Finding the TP, TN, FP, FN
    #                TP_temp, FP_temp, TN_temp, FN_temp = \
    #                perf_measure(y_test,y_pred)
    #
    #                # Summing up the values for each iteration
    #                TP[s][a] += TP_temp
    #                FP[s][a] += FP_temp
    #                TN[s][a] += TN_temp
    #                FN[s][a] += FN_temp
    #
    #
    #            ####################################################
    #            #------------- performance measurement ------------#
    #            ####################################################
    #
    #            accuracy[s][a] = (TP[s][a]+TN[s][a])/(TP[s][a]+TN[s][a]+FP[s][a]+FN[s][a])
    #            recall[s][a] = TP[s][a]/(TP[s][a]+FN[s][a])
    #            specifity[s][a] = TN[s][a]/(TN[s][a]+FP[s][a])
    #            precision[s][a] = TP[s][a]/(TP[s][a]+FP[s][a])
    #            blncd_aqrc[s][a] = (recall[s][a]+specifity[s][a])/2
    #            f1[s][a] = (2*recall[s][a]*precision[s][a])/(recall[s][a]+precision[s][a])
    #
    #            # Transforming dicts of dicts to dataframes
    #            f1_df = pd.DataFrame.from_dict(f1)
    #            # Saving the dataframe to csv file
    #            f1_df.to_csv('./results/f1_score.csv')
    #
    #            recall_df = pd.DataFrame.from_dict(recall)
    #            recall_df.to_csv('./results/recall.csv')
    #
    #            precision_df = pd.DataFrame.from_dict(precision)
    #            precision_df.to_csv('./results/precision.csv')
    #
    #            accuracy_df = pd.DataFrame.from_dict(accuracy)
    #            accuracy_df.to_csv('./results/accuracy.csv')
    #
    #            blncd_aqrc_df = pd.DataFrame.from_dict(blncd_aqrc)
    #            blncd_aqrc_df.to_csv('./results/balanced_accuracy.csv')
    #
    #            specifity_df = pd.DataFrame.from_dict(specifity)
    #            specifity_df.to_csv('./results/specifity.csv')

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
    specifity = {}
    accuracy = {}
    precision = {}
    blncd_aqrc = {}
    f1 = {}

    # Defining the same variables for Late Fusion Average (LFA)
    TP_LFA = {}
    TN_LFA = {}
    FP_LFA = {}
    FN_LFA = {}
    recall_LFA = {}
    specifity_LFA = {}
    accuracy_LFA = {}
    precision_LFA = {}
    blncd_aqrc_LFA = {}
    f1_LFA = {}

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

        TP_LFA[a] = 0
        TN_LFA[a] = 0
        FP_LFA[a] = 0
        FN_LFA[a] = 0
        recall_LFA[a] = 0
        specifity_LFA[a] = 0
        accuracy_LFA[a] = 0
        precision_LFA[a] = 0
        blncd_aqrc_LFA[a] = 0
        f1_LFA[a] = 0

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
        for i in range(num_folds):  # Looping over number of folds
            print(i)
            m = 0  # this is for tracking the sensors
            # Getting the test and train dataset
            train_dataset, test_dataset = train_test_split(i, num_folds, cvdir, dataset)

            for s in Sensor:
                print(s)
                X_train, y_train, X_test, y_test, X_test_f, y_test_f = pred_resp(
                    train_dataset, test_dataset, s, a, all_sensors_f
                )

                if m == 0:
                    y_prob = np.zeros([len(y_test_f), len(all_sensors_f)])

                # NOTE: random_state=0 and solver="lbfgs" are unnecessary
                clf = LogisticRegression(
                    random_state=0, class_weight="balanced", solver="lbfgs"
                )
                clf = clf.fit(X_train, y_train)

                # Removing train and test dataset to free up some space
                #                del(train_dataset,test_dataset)

                y_prob[:, m] = clf.predict_proba(X_test_f)[:, 1]
                y_pred = clf.predict(X_test)
                # NOTE: Fixed bug in original code
                y_test = y_test.values

                TP_temp, FP_temp, TN_temp, FN_temp = perf_measure(y_test, y_pred)

                TP[a][s] += TP_temp
                FP[a][s] += FP_temp
                TN[a][s] += TN_temp
                FN[a][s] += FN_temp

                m = m + 1

            # Applyingthe late fusion average
            y_pred_LFA = LFA(y_prob)
            y_test_f = np.array(y_test_f)
            TP_temp, FP_temp, TN_temp, FN_temp = perf_measure(y_test_f, y_pred_LFA)

            TP_LFA[a] += TP_temp
            FP_LFA[a] += FP_temp
            TN_LFA[a] += TN_temp
            FN_LFA[a] += FN_temp

    ####################################################
    # ------------- performance measurement ------------#
    ####################################################

    # Knowing the TP, TN, FP, FN for each pair of activity-sensor, it is
    # a straightforward task to compute the metrics...
    for a in activity:
        accuracy_LFA[a] = (TP_LFA[a] + TN_LFA[a]) / (
            TP_LFA[a] + TN_LFA[a] + FP_LFA[a] + FN_LFA[a]
        )
        recall_LFA[a] = TP_LFA[a] / (TP_LFA[a] + FN_LFA[a])
        specifity_LFA[a] = TN_LFA[a] / (TN_LFA[a] + FP_LFA[a])
        precision_LFA[a] = TP_LFA[a] / (TP_LFA[a] + FP_LFA[a])
        blncd_aqrc_LFA[a] = (recall_LFA[a] + specifity_LFA[a]) / 2
        f1_LFA[a] = (2 * recall_LFA[a] * precision_LFA[a]) / (
            recall_LFA[a] + precision_LFA[a]
        )

        for s in Sensor:
            accuracy[a][s] = (TP[a][s] + TN[a][s]) / (
                TP[a][s] + TN[a][s] + FP[a][s] + FN[a][s]
            )
            recall[a][s] = TP[a][s] / (TP[a][s] + FN[a][s])
            specifity[a][s] = TN[a][s] / (TN[a][s] + FP[a][s])
            precision[a][s] = TP[a][s] / (TP[a][s] + FP[a][s])
            blncd_aqrc[a][s] = (recall[a][s] + specifity[a][s]) / 2
            f1[a][s] = (2 * recall[a][s] * precision[a][s]) / (
                recall[a][s] + precision[a][s]
            )

    # Transforming dicts to dataframes
    f1_ss = pd.DataFrame.from_dict(f1)
    f1_ss = f1_ss.transpose()
    f1_LFA = pd.Series(f1_LFA)
    f1_LFA = pd.DataFrame(f1_LFA, columns=["LFA"])
    f1 = pd.concat([f1_ss, f1_LFA], axis=1)
    # Saving the dataframe to csv file
    f1.to_csv("./results/f1_score.csv")

    recall_ss = pd.DataFrame.from_dict(recall)
    recall_ss = recall_ss.transpose()
    recall_LFA = pd.Series(recall_LFA)
    recall_LFA = pd.DataFrame(recall_LFA, columns=["LFA"])
    recall = pd.concat([recall_ss, recall_LFA], axis=1)
    recall.to_csv("./results/recall.csv")

    precision_ss = pd.DataFrame.from_dict(precision)
    precision_ss = precision_ss.transpose()
    precision_LFA = pd.Series(precision_LFA)
    precision_LFA = pd.DataFrame(precision_LFA, columns=["LFA"])
    precision = pd.concat([precision_ss, precision_LFA], axis=1)
    precision.to_csv("./results/precision.csv")

    accuracy_ss = pd.DataFrame.from_dict(accuracy)
    accuracy_ss = accuracy_ss.transpose()
    accuracy_LFA = pd.Series(accuracy_LFA)
    accuracy_LFA = pd.DataFrame(accuracy_LFA, columns=["LFA"])
    accuracy = pd.concat([accuracy_ss, accuracy_LFA], axis=1)
    accuracy.to_csv("./results/accuracy.csv")

    blncd_aqrc_ss = pd.DataFrame.from_dict(blncd_aqrc)
    blncd_aqrc_ss = blncd_aqrc_ss.transpose()
    blncd_aqrc_LFA = pd.Series(blncd_aqrc_LFA)
    blncd_aqrc_LFA = pd.DataFrame(blncd_aqrc_LFA, columns=["LFA"])
    blncd_aqrc = pd.concat([blncd_aqrc_ss, blncd_aqrc_LFA], axis=1)
    blncd_aqrc.to_csv("./results/blncd_aqrc.csv")

    specifity_ss = pd.DataFrame.from_dict(specifity)
    specifity_ss = specifity_ss.transpose()
    specifity_LFA = pd.Series(specifity_LFA)
    specifity_LFA = pd.DataFrame(specifity_LFA, columns=["LFA"])
    specifity = pd.concat([specifity_ss, specifity_LFA], axis=1)
    specifity.to_csv("./results/specifity.csv")
