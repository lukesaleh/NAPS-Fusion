# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:17:55 2019

@author: 14342
"""

from __future__ import absolute_import, division, print_function
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, "../examples")
sys.path.append(parent_dir)
from pyds_local import *
from Naive_Adaptive_Sensor_Fusion import *
import config


from timeit import default_timer as timer
import random
import glob
import multiprocessing.pool as mp
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings
from imblearn.over_sampling import SMOTE
import timeit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from os.path import join as pjoin

import dill
from multiprocessing import Pool, TimeoutError

# from pathos.multiprocessing import ProcessingPool as Pool
# from multiprocess import Pool

from pycm import ConfusionMatrix

# import tensorflow as tf

# caution: path[0] is reserved for script path (or '' in REPL)
from ucsd_fusion import save_model_to_pickle as save_var_to_pickle
from ucsd_fusion import load_model_from_pickle as load_var_from_pickle
from not_my_code import get_performance_metrics, plot_confusion_matrix

warnings.filterwarnings("ignore")



def readdata_csv(data_dir):
    """This function gets the directory of the datasets and returns the dataset
    containing information of all 60 users

    Input:
        data_dir[string]: holds the directory of all the csv files (60)

    Output:
        grand_dataset[dict]: a dictionary of all the users' data. The format is:
            grand_dataset:{'uuid1': dataframe of csv file of the user 1
                           'uuid2': dataframe of csv file of the user 2
                           'uuid3': dataframe of csv file of the user 3
                           ...}
    """
    length_uuids = 36  # number of characters for each uuid
    data_list = glob.glob(os.path.join(os.getcwd(), data_dir, "*.csv"))
    # grand_dataset is a dict. that holds the uuids and correspondong datast
    grand_dataset = {}
    for i in range(len(data_list)):
        #    for i in range(5):
        # dismantles the file name and picks only uuids (first 36 characters)
        uuid = os.path.basename(data_list[i])[:length_uuids]
        dataset_ith = pd.read_csv(data_list[i])
        print(
            "User {}/{}  -> Shape of the data     {}".format(
                i + 1, len(data_list), dataset_ith.shape
            )
        )
        grand_dataset[uuid] = dataset_ith
    return grand_dataset


def Set_Act_Sens():
    """This function defines two dictionaries for activities and sensors. Each
    dictionaray holds the the range of columns for the specified sensor or
    activity.

    Input:
    Output:
        Activities[dict]: a dictionary of the activities and their corresponding
                        column number
        Sensors[dict]: a dictionary of the sensors and their corresponding range
                        of features
    """
    Activities = {}
    Activities["label:LYING_DOWN"] = 226
    Activities["label:SITTING"] = 227
    Activities["label:FIX_walking"] = 228
    Activities["label:FIX_running"] = 229
    Activities["label:BICYCLING"] = 230
    Activities["label:SLEEPING"] = 231
    Activities["label:OR_standing"] = 270

    Sensors = {}
    Sensors["Acc"] = list(range(1, 27))
    Sensors["Gyro"] = list(range(27, 53))
    #    Sensors['Mag'] = list(range(53,84))
    Sensors["W_acc"] = list(range(84, 130))
    #    Sensors['Compass'] = list(range(130,139))
    Sensors["Loc"] = list(range(139, 156))
    Sensors["Aud"] = list(range(156, 182))
    #    Sensors['AP'] = list(range(182,184))
    Sensors["PS"] = list(np.append(range(184, 210), range(218, 226)))
    #    Sensors['LF'] = list(range(210,218))

    return (Activities, Sensors)


def Response_Merger(data, cols_to_merge):
    """
    This function takes in the dataset and a list of columns of different labels
    to merge and combnine them using a logical OR to give back one column. ex.
    l1+l2+l3 -> {l1,l2,l3}

    Parameters
    ----------
    data : dataframe
        dataframe of the dataset (ex. training data).
    cols_to_merge : list
        a list of the columns to merge with a logical OR. like:
        ['Lying_down','Sleeping'].

    Returns
    -------
    merged_label: dataframe
        a dataframe with only one column whose values are binary

    """
    data = data[cols_to_merge].fillna(0)

    merged_label = data[cols_to_merge[0]]

    for i in range(1, len(cols_to_merge)):
        merged_label = np.logical_or(merged_label, data[cols_to_merge[i]]) * 1

    merged_label = merged_label.to_frame()
    merged_label.columns = ["Merged_label"]

    #    col_name = ''.join(cols_to_merge[:])
    #    cols_to_merge = add_label(cols_to_merge)
    #
    #    # First we impute the NaN with 0 in all the columns that are about to be merged
    #    data = data[cols_to_merge].fillna(0)
    #
    #    # Now find the logical OR of the desired columns (labels)
    #    merged_label = data[cols_to_merge[0]]
    #    merged_label.name = col_name
    #
    #    for i in range(len(cols_to_merge)):
    #        merged_label = np.logical_or(merged_label, data[cols_to_merge[i]])
    return merged_label


def Xy(
    data,
    feature_sets,
    feature_set_idx,
    response_perms_1,
    response_perms_2,
    response_perm_idx,
    impute=False,
):
    """This function takes data, feature sets matrix, respnse perms, the index
    of the row of the desired feature set, and the index of the desired rows
    of the response variables and gives back the X and y"""
    # Maybe add sorting later
    # Maybe some more manupulations on response variable
    X = data.iloc[:, list(feature_sets[feature_set_idx, :])]
    #    X.loc[:,:] = preprocessing.scale(X)

    if impute is not False:
        X = X.fillna(0)

    y1 = Response_Merger(data, response_perms_1[response_perm_idx])
    y2 = Response_Merger(data, response_perms_2[response_perm_idx])

    aux = y1 + y2
    indices = np.where(aux == 1)[0]

    if len(indices) > 1:
        if config.Using_UCSD == False:
            X = X.iloc[indices, :].reset_index(drop=True)
            y1 = y1.iloc[indices, :].reset_index(drop=True)
            y2 = y2.iloc[indices, :].reset_index(drop=True)
        else:
            y1 = y1.loc[indices, :]
            y2 = y2.loc[indices, :]
            X = X.loc[indices, :]

    y = y1

    xy = pd.concat([X, y], axis=1)
    xy = xy.dropna()

    y = xy.iloc[:, -1]
    X = xy.iloc[:, 0:-1]

    return (X, y, y1, y2)


def train_test_spl(test_fold, num_folds, fold_dir, grand_dataset):
    """This function takes the number of test fold (ranging from 0 to 4) and
    number of folds (in this case 5) and directory where the folds' uuids are
    and the dataset, and returns train and test datasets

    Input:
        test_fold_idx[integer]: an integer indicating the index of the test fold
        fold_dir[string]: holds the directory in which the folds' uuids are
        grand_dataset[dict]: a dictionary of all users' data. (essentially the
                             output of readdata_csv())
    Output:
        train_dataset[pandas.dataframe]: dataframe of the train dataset
        test_dataset[pandas.dataframe]: dataframe of the test dataset
    """
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
        train_dataset = pd.concat(
            [train_dataset, grand_dataset[i[0]]], axis=0, ignore_index=True
        )

    for j in test_uuids:
        test_dataset = pd.concat(
            [test_dataset, grand_dataset[j[0]]], axis=0, ignore_index=True
        )

    return (train_dataset, test_dataset)


def train_test_split_new(full_dataset_directory):

    # Load the dataset from the directory
    data = pd.read_csv(full_dataset_directory)
    
    # Split the dataset into train and test sets with test size of 0.2 and random state of 42
    train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=42)

    return (train_dataset, test_dataset)



def get_folds_uuids(fold_dir):
    """
    The function gets the directory where the the folds text files are located
    and returns a list of five np.arrays in each of them the uuids of the
    corresponding fold are stored.

    Input:
        fold_dir[string]: holds the directory in which folds are

    Output:
        folds_uuids[list]: a list of numpy arrays. Each array holds the uuids
                    in that fold. ex.
                    folds_uuids = [('uuid1','uuid2',...,'uuid12'),
                                   ('uuid13','uuid14',...,'uuid24'),
                                   ...,
                                   ('uuid49','uuid50',...,'uuid60')]
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


def train_NAPS_Models(
    train_dataset,
    feature_sets,
    j,
    Response_Perm_1,
    Response_Perm_2,
    i,
    bagging_R,
    num_bags,
    seed_number,
    impute=True,
):
    t0 = timeit.default_timer()
    X_train, y_train, y1, y2 = Xy(
        train_dataset, feature_sets, j, Response_Perm_1, Response_Perm_2, i, impute=True
    )

    t1 = timeit.default_timer()
    # print('\tgetting training data took : ', t1-t0)

    smt = SMOTE()

    #FIXME: Add SMOTE onto the augmented data
    #NOTE: Need to look at what Napoli did in the paper
    X_train_tmp, y_train_tmp = smt.fit_resample(X_train, y_train)
    t2 = timeit.default_timer()
    # print('\tSMOTE took : ', t2-t1)

    X_train = pd.DataFrame(X_train_tmp, columns=X_train.columns)
    y_train = pd.DataFrame(y_train_tmp, columns=["Merged_label"])

    U2 = Uncertainty_Bias([y1, y2])

    t3 = timeit.default_timer()
    # create a model and train it
    NAPS_sample = DS_Model(Response_Perm_1[i], Response_Perm_2[i], X_train, y_train, j)
    t4 = timeit.default_timer()
    # print('\tCreating the DS model took : ', t4-t3)

    NAPS_sample.Bags_Trainer(
        X_train, y_train, bagging_R, num_bags, config.random_seed_number
    )
    t5 = timeit.default_timer()
    # print('\tTraining bags took : ', t5-t4)

    NAPS_sample.Uncertainty_B = U2

    return NAPS_sample


def run_dill_encoded(payload):
    fun = dill.loads(payload[0])
    return fun(payload[1])


def main():
    # =============================================================================#
    # --------------------------| Tensorflow for LR |------------------------------#
    # =============================================================================#

    # num_classes = 10 # 0 to 9 digits
    # # num_features = 784 # 28*28
    # learning_rate = 0.01
    # training_steps = 1000
    # batch_size = 256
    # display_step = 50

    # W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
    # b = tf.Variable(tf.zeros([num_classes]), name="bias")

    # def logistic_regression(x):
    #     # Apply softmax to normalize the logits to a probability distribution.
    #     return tf.nn.softmax(tf.matmul(x, W) + b)

    # def cross_entropy(y_pred, y_true):
    #     # Encode label to a one hot vector.
    #     y_true = tf.one_hot(y_true, depth=num_classes)
    #     # Clip prediction values to avoid log(0) error.
    #     y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    #     # Compute cross-entropy.
    #     return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

    # def accuracy(y_pred, y_true):
    #     # Predicted class is the index of the highest score in prediction vector (i.e. argmax).
    #     correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    #     return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # # Stochastic gradient descent optimizer.
    # optimizer = tf.optimizers.SGD(learning_rate)

    if config.Using_UCSD == True:

        # =============================================================================#
        # -------------------------------| INPUTS |------------------------------------#
        # =============================================================================#
        start_time = timeit.default_timer()

        # =============================================================================#
        # -------------------------| Reading in the data |-----------------------------#
        # =============================================================================#

        Activities, Sensors = Set_Act_Sens()  # creating two dicts for sensor and activity

        print("\n#------------- Reading in the Data of Users -------------#\n")
        if not os.path.exists(config.data_dir):
            print('Data directory does not exist!')
            sys.exit(1)

        dataset_uuid = readdata_csv(
            config.data_dir
        )  # reading all data and storing in "dataset" a DF
        stop1 = timeit.default_timer()

        print("Reading the data took:   ", int(stop1 - start_time))

        uuids = list(dataset_uuid.keys())

        print("\n#-------------- Combining the Data of Users-------------#\n")

        # We concatenate the data of all participants (60) to get one dataset for all
        dataset_ag = dataset_uuid[
            uuids[0]
        ]  # "dataset_ag" is the aggregation of all user's data
        for i in range(1, len(uuids)):
            dataset_ag = pd.concat(
                [dataset_ag, dataset_uuid[uuids[i]]], axis=0, ignore_index=True
            )

        dataset_ag.iloc[:, config.feature_range] = preprocessing.scale(
            dataset_ag.iloc[:, config.feature_range]
        )
        stop2 = timeit.default_timer()

        print("Combining the data took:   ", int(stop2 - stop1))
        if os.path.exists('combined_data.csv') is not True:
            dataset_ag.to_csv('combined_data.csv', index=False)
        # =============================================================================#
        # -----------------------------| DST Setups |----------------------------------#
        # =============================================================================#

        # We create feature sets, a sample mass function (initialized to 0) and response
        # permutations 1 and 2 in which corresponding elements are exclusive and exhaustive

        feature_sets = feature_set(
            config.sensors_to_fuse,
            config.feature_sets_st,
            Sensors,
            config.random_seed_number,
            feat_set_count=config.feature_sets_count,
        )
        mass_template = BPA_builder(config.FOD)
        Response_Perm_1, Response_Perm_2 = pair_resp(mass_template)

        num_p = len(config.FOD)
        num_fs = len(feature_sets)
        num_rp = len(Response_Perm_1)
        num_folds = 5

        # smt = SMOTE()

        # find the train_dataset
        # at personal level:
        print("\n#-------------- Obtaining Training Dataset -------------#\n")

        # TODO: Do they only look at 1 fold? That seems to be the case
        train_dataset, test_dataset = train_test_spl(
            0, num_folds, config.cvdir, dataset_uuid
        )

        stop3 = timeit.default_timer()
        print("Obtaining the training dataset took:   ", int(stop3 - stop2))

        print("Training dataset has  {}  samples".format(len(train_dataset)))

    else:

        # =============================================================================#
        # -------------------------------| INPUTS |------------------------------------#
        # =============================================================================#
        start_time = timeit.default_timer()

        # =============================================================================#
        # -------------------------| Reading in the data |-----------------------------#
        # =============================================================================#
        Activities, Sensors = config.Set_Act_Sens_NEW()  # creating two dicts for sensor and activity

        #stop2 = timeit.default_timer()

        #print("Combining the data took:   ", int(stop2 - stop1))

        # =============================================================================#
        # -----------------------------| DST Setups |----------------------------------#
        # =============================================================================#

        # We create feature sets, a sample mass function (initialized to 0) and response
        # permutations 1 and 2 in which corresponding elements are exclusive and exhaustive

        feature_sets = feature_set(
            config.sensors_to_fuse,
            config.feature_sets_st,
            Sensors,
            config.random_seed_number,
            feat_set_count=config.feature_sets_count,
        )
        mass_template = BPA_builder(config.FOD)
        Response_Perm_1, Response_Perm_2 = pair_resp(mass_template)

        num_p = len(config.FOD)
        num_fs = len(feature_sets)
        num_rp = len(Response_Perm_1)
        num_folds = 5

        # smt = SMOTE()

        # find the train_dataset
        # at personal level:
        print("\n#-------------- Obtaining Training Dataset -------------#\n")

        # TODO: Do they only look at 1 fold? That seems to be the case
        train_dataset, test_dataset = train_test_split_new(config.data_dir)

        stop3 = timeit.default_timer()
        print("Obtaining the training dataset took:   ", int(stop3 - start_time))

        print("Training dataset has  {}  samples".format(len(train_dataset))) 

    # =============================================================================#
    # ------------------| Creating and Training all the models |-------------------#
    # =============================================================================#

    print("\n#-------------- Creating and Training Models -------------#\n")

    # ------------------------ Parallelization goes here  -------------------------#
    # Parallelization with 7 processes takes 16 mins compared to 45 min
    # single-threaded

    model_filename = pjoin(config.results_dir, "fold_9_models.pkl")

    if os.path.isfile(model_filename):
        print("(Skipping model training)")
        NAPS_models = load_var_from_pickle(model_filename)
    else:
        if config.parallelize:
            impute = True
            NAPS_models_dict = {}

            pool = mp.Pool(processes=config.num_prc)

            # Callback function
            def save_result_cb(result):
                # result[0] is the response permutation number and result[1] is the
                # model
                NAPS_models_dict.update({result[0]: result[1]})

            # Create threads
            for i_rp in range(num_rp):
                pool.apply_async(
                    NAPS_Models_Trainer,
                    args=(
                        i_rp,
                        range(num_fs),
                        train_dataset,
                        feature_sets,
                        Response_Perm_1,
                        Response_Perm_2,
                        config.bagging_R,
                        config.num_bags,
                        config.random_seed_number,
                        impute,
                    ),
                    callback=save_result_cb,
                )

            pool.close()
            pool.join()

            # TODO: Turn dictionary to list
            NAPS_models = [NAPS_models_dict[ii] for ii in range(num_rp)]
        else:
            NAPS_models = []
            print("\nLooping over Response Permutations \n ")

            for i in range(num_rp):  # i runs over response permutations
                start_rp = timer()

                print("\nResponse Permutation {}/{}".format(i + 1, num_rp))
                print("\n\tLooping over feature sets")
                NAPS_models.append([])

                progress = ProgressBar(num_fs, fmt=ProgressBar.FULL)

                for j in range(num_fs):  # j runs over feature sets
                    NAPS_models[i].append([])

                    NAPS_models[i][j] = train_NAPS_Models(
                        train_dataset,
                        feature_sets,
                        j,
                        Response_Perm_1,
                        Response_Perm_2,
                        i,
                        config.bagging_R,
                        config.num_bags,
                        config.random_seed_number,
                        impute=True,
                    )

                    # find X and y
                    # NOTE: This function call is already in train_NAPS_Models
                    # X_train, y_train, y1, y2 = Xy(
                    #     train_dataset,
                    #     feature_sets,
                    #     j,
                    #     Response_Perm_1,
                    #     Response_Perm_2,
                    #     i,
                    #     config.random_seed_number,
                    #     impute=True,
                    # )
                    # #        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                    #
                    # X_train_tmp, y_train_tmp = smt.fit_sample(X_train, y_train)
                    #
                    # X_train = pd.DataFrame(X_train_tmp, columns=X_train.columns)
                    # y_train = pd.DataFrame(y_train_tmp, columns=["Merged_label"])
                    #
                    # if j == 0:
                    #     U2 = Uncertainty_Bias([y1, y2])
                    #
                    # # create a model and train it
                    # NAPS_models[i][j] = DS_Model(
                    #     Response_Perm_1[i], Response_Perm_2[i], X_train, y_train, j
                    # )
                    # NAPS_models[i][j].Bags_Trainer(X_train, y_train, config.bagging_R, config.num_bags)
                    # NAPS_models[i][j].Uncertainty_B = U2

                    progress.current += 1
                    progress()

                progress.done()

                stop_rp = timer()

                print("\n It took : ", stop_rp - start_rp)

        # Save models
        save_var_to_pickle(model_filename, NAPS_models)

    stop4 = timeit.default_timer()
    print("Training all models took:   ", int(stop4 - stop3))

    # =============================================================================#
    # ------------------| Model Selection and Testing Models |---------------------#
    # =============================================================================#

    print("\n#-------------- Testing the Models -------------#\n")
    test_dataset[config.FOD] = test_dataset[config.FOD].fillna(0)
    test_dataset = test_dataset[np.sum(test_dataset[config.FOD], axis=1) != 0]

    y_test_ag = np.zeros([len(test_dataset), len(config.FOD)])
    y_pred_ag = np.zeros([len(test_dataset), len(config.FOD)])
    #FIXME: fill with NaN, then remove all NaNs at end
    model_auc_ag = np.zeros([len(test_dataset), num_rp*num_fs])
    model_pred_ag2 = np.zeros([len(test_dataset), num_rp*num_fs])
    model_true_ag = np.zeros([len(test_dataset), num_rp*num_fs ])

    num_test_points = len(test_dataset)
    print(f"Test dataset has {num_test_points} samples")

    if config.parallelize:
        def Fun(idx):
            import numpy as np
            import config
            from main import Xy
            from sklearn.metrics import roc_curve, auc
            from Naive_Adaptive_Sensor_Fusion import Model_Selector, Fuse_and_Predict
            # print(idx, "/", len(test_dataset))
            test_sample = test_dataset.iloc[idx, :]
            test_sample = test_sample.to_frame().transpose()
            y_test_single = np.floor(test_sample[config.FOD].fillna(0).values)

            assert np.sum(y_test_single) == 1

            Uncertainty_Mat = np.ones([num_rp, num_fs])
            auc_scores = []
            y_model_pred = []
            y_true = []
            for i in range(num_rp):
                for j in range(num_fs):
                    X_test, y_test, y1, y2 = Xy(
                        test_sample,
                        feature_sets,
                        j,
                        Response_Perm_1,
                        Response_Perm_2,
                        i,
                        impute=True,
                    )
                    if len(X_test) != 0 or len(y_test) != 0:
                    #    #TODO: try larger number of bags to see influence on votes
                    #    Uncertainty_Mat[i][j] = (
                    #         NAPS_models[i][j].Uncertainty_B
                    #         + NAPS_models[i][j].Uncertainty_Context(X_test, y_test)
                    #     ) / 2
                        theta1 = NAPS_models[i][j].Uncertainty_B
                        theta2 = NAPS_models[i][j].Uncertainty_Context(X_test, y_test)
                        param = 0.5 # Hyperparameter
                        d = (np.exp(1 - param) - 1) ** np.e
                        Uncertainty_Mat[i][j] = (1 - np.exp(-0.5 * (theta1 + theta2) / d)) / (1 - np.exp(-1 / d))
                    NAPS_models[i][j].Mass_Function_Setter(
                        Uncertainty_Mat[i][j], X_test
                    )
                    y_true.append(NAPS_models[i][j].actual_preds)
                    y_model_pred.append(NAPS_models[i][j].test_inputs)
            #flattened_list = [item for sublist in y_model_pred for item in sublist]
            #flattened_true = [item for sublist in y_true for item in sublist]
            

            # =========\ Model Selection /==========#

            Selected_Models_idx = Model_Selector(
                Uncertainty_Mat, config.models_per_rp, num_rp, 1
            )
            y_pred_single = Fuse_and_Predict(
                Selected_Models_idx,
                NAPS_models,
                config.FOD,
                num_p,
                num_rp,
                config.models_per_rp,
            )

            return y_test_single, y_pred_single, y_model_pred, y_true 

        # defaults to number of available CPU's
        test_pool = Pool()
        # this may take some guessing ... take a look at the docs to decide
        chunk_size = 20

        t_pool_start = timeit.default_timer()
        encoded_fun = dill.dumps(Fun)

        #TODO: try catch on the imap, see values and types for imap and the stuff withing
        #TODO: when seeing exception, pause the program

        # try:
        #     enumerate(test_pool.imap(
        #         run_dill_encoded,
        #         zip(num_test_points * [encoded_fun], range(num_test_points)),
        #         chunk_size,
        #     ))
        
        # except TypeError:
        #     print('Type error achieved')
        #     print(type(test_pool.imap(
        #         run_dill_encoded,
        #         zip(num_test_points * [encoded_fun], range(num_test_points)),
        #         chunk_size,
        #     )))

        for ind, res in enumerate(
            test_pool.imap(
                run_dill_encoded,
                zip(num_test_points * [encoded_fun], range(num_test_points)),
                chunk_size,
            )
        ):
            y_test_ag[ind] = res[0]
            y_pred_ag[ind] = res[1]
            model_pred_ag2[ind] = res[2]
            model_true_ag[ind] = res[3]

        print(timeit.default_timer() - t_pool_start)
    else:
        y_model_pred = []
        y_true = []
        for t in range(0, len(test_dataset)):
            print(t, "/", len(test_dataset))
            test_sample = test_dataset.iloc[t, :]
            test_sample = test_sample.to_frame().transpose()
            y_test_ag[t, :] = np.floor(test_sample[config.FOD].fillna(0).values)

            # TODO: I think this assertion is wrongly written
            assert np.sum(y_test_ag == 1)

            Uncertainty_Mat = np.ones([num_rp, num_fs])
            
            for i in range(num_rp):
                for j in range(num_fs):
                    X_test, y_test, y1, y2 = Xy(
                        test_sample,
                        feature_sets,
                        j,
                        Response_Perm_1,
                        Response_Perm_2,
                        i,
                        impute=True,
                    )
                    if len(X_test) != 0 or len(y_test) != 0:
                        # Uncertainty_Mat[i][j] = (
                        #     NAPS_models[i][j].Uncertainty_B
                        #     + NAPS_models[i][j].Uncertainty_Context(X_test, y_test)
                        # ) / 2
                        theta1 = NAPS_models[i][j].Uncertainty_B
                        theta2 = NAPS_models[i][j].Uncertainty_Context(X_test, y_test)
                        p = 0.5 # Hyperparameter
                        d = (np.exp(1 - p) - 1) ** np.e
                        Uncertainty_Mat[i][j] = (1 - np.exp(-0.5 * (theta1 + theta2) / d)) / (1 - np.exp(-1 / d))
                    NAPS_models[i][j].Mass_Function_Setter(
                        Uncertainty_Mat[i][j], X_test
                    )
                    # NAPS_models[i][j].Mass_Function_Printer()
                    y_true.append(NAPS_models[i][j].actual_preds)
                    y_model_pred.append(NAPS_models[i][j].test_inputs)
                    #auc_score.append(NAPS_models[i][j].Model_AUC())
            
            
            # =========\ Model Selection /==========#
            Selected_Models_idx = Model_Selector(
                Uncertainty_Mat, config.models_per_rp, num_rp, 1
            )
            fusion_result = Fuse_and_Predict(
                Selected_Models_idx,
                NAPS_models,
                config.FOD,
                num_p,
                num_rp,
                config.models_per_rp,
            )
            y_pred_ag[t, :] = fusion_result
            
    stop5 = timeit.default_timer()
    print("Testing took:   ", int(stop5 - stop4))

    y_test_vector = np.argmax(y_test_ag, axis=1)
    y_pred_vector = np.argmax(y_pred_ag, axis=1)
    model_auc_vector = model_auc_ag
    if config.parallelize:
        model_auc_vector = [row[0] for row in model_auc_ag]
        model_true_flat = [item for sublist in model_true_ag for item in sublist ]
        model_pred_flat = [item for sublist in  model_pred_ag2 for item in sublist]
        fpr, tpr, threshold = roc_curve(model_true_flat, model_pred_flat)   
        if config.model_used == "logistic regression":
            print('AUC for logistic regression models in bags:', auc(fpr, tpr)) #TODO: save to pickle
        else:
            print('AUC for decision tree models in bags:', auc(fpr, tpr))
        
    # conf_mat = confusion_matrix(y_test_ag, y_pred_ag)
    # accuracy = accuracy_score(y_test_ag, y_pred_ag)
    # balanced_accuracy = balanced_accuracy_score(y_test_ag, y_pred_ag)
    # f1 = f1_score(y_test_ag, y_pred_ag)
    else:
        fpr, tpr, threshold = roc_curve(y_true, y_model_pred)
        if config.model_used == "logistic regression":
            print('AUC for logistic regression models in bags:',auc(fpr, tpr)) #TODO: Save to pickle
        else:
            print('AUC for decision tree models in bags:',auc(fpr, tpr))
    # TODO: Automate this
    # pretty_labels =  {idx: get_pretty_label_name(raw_label) for idx, raw_label in enumerate(config.FOD)}
    cm = ConfusionMatrix(actual_vector=y_test_vector, predict_vector=y_pred_vector)
    cm.relabel(mapping=config.pretty_labels)
    print(cm)
    # print('Best model AUC:',max(model_auc_vector))
    # print('Worst model AUC:',min(model_auc_vector))
    # print('Average model AUC',np.nanmean(model_auc_vector))
    plot_confusion_matrix(
        y_test_vector,
        y_pred_vector,
        list(config.pretty_labels.values()),
        os.path.join(config.results_dir, "figures"),
    )

    cm.save_csv(
        os.path.join(config.results_dir, "performance_metrics"),
        matrix_save=True,
        # Add class labels to confusion matrix
        header=True,
    )

    debug_point = []

    #!!!!!!!!! Model Selection based on the uncertainty should be fixed !!!!!!!!


if __name__ == "__main__":
    main()
