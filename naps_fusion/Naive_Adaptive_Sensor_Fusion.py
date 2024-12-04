# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:13:08 2019

@author: 14342
"""

from pyds_local import *
from sklearn.metrics import roc_auc_score, roc_curve, auc
import random
import numpy as np
import sys
import pandas as pd
import config
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import itertools
import sys
import re
from imblearn.over_sampling import SMOTE


class DS_Model:
    def __init__(self, resp_var_1, resp_var_2, X_train, y_train, feature_set_idx):
        """
        Initializing a DS model

        Parameters
        ----------
        resp_var_1 : list
        resp_var_2 : list
            These two are the response variables that we want to build a model
            on. For example if we are building a model to classify {c1} vs {c2,c3}
            then [c1] would be our resp_var_1 and [c2,c3] would be our resp_var_2.

        X_train : pd.dataframe
            training features in a dataframe format.
        y_train : pd.dataframe
            training labels as a vector.
        feature_set_idx : integer
            Index of the feature set (out of all the randomly created feature sets).

        Returns
        -------
        None.

        """
        self.Feature_Set_idx = feature_set_idx
        self.Response_Variables = [resp_var_1, resp_var_2]

        if config.model_used == "decision tree":
            self.clf = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=5, max_depth=3).fit(X_train, y_train)
        elif config.model_used == "logistic regression":
            self.clf = LogisticRegression(
            class_weight="balanced", solver="saga", n_jobs=-1
            ).fit(X_train, y_train)
        else:
            print('Invalid model selected!')
            sys.exit(1)
        self.Bags = []  # list of the bags for this model
        self.Uncertainty_B = 0  # Uncertainty of the biased model
        self.mass = MassFunction()  # Mass function of the model
        self.test_inputs = 0
        self.actual_preds = 0

    def Mass_Function_Setter(self, uncertainty, X):
        """
        We used pyds package (a dempster shafer package) to define the mass function
        given the probabilities and uncertainty.
        """
        probability = self.clf.predict_proba(X) 
        self.mass[self.Response_Variables[0]] = probability[0, 0] * (1 - uncertainty)
        self.mass[self.Response_Variables[1]] = probability[0, 1] * (1 - uncertainty)
        #Needs exponentiation formula from 4.2.3

        #TODO: Check if it sums to 1-uncertainty
    def Mass_Function_Printer(self):
        print(self.mass)
    
    def Bags_Trainer(self, X_train, y_train, ratio, num_bags, seed_number):
        """
        This function trains bags

        Parameters
        ----------
        X_train : pd.dataframe
            training features in a dataframe format.
        y_train : pd.dataframe
            training labels as a vector.
        ratio : float
            Ratio of the generated bagging size to the actual training dataset.
        num_bags : integer
            number of the bags to generate.

        Returns
        -------
        None.

        """
        # See the PRNG
        random.seed(a=seed_number)

        for i in range(num_bags):
            self.Bags.append(clone(self.clf))
            indices = random.choices(
                list(range(len(X_train))), k=int(ratio * (len(X_train)))
            )
            X_train_Bag = X_train.iloc[indices, :]
            y_train_Bag = y_train.iloc[indices]

            self.Bags[i].fit(X_train_Bag, y_train_Bag)

    def Model_AUC(self):
        fpr, tpr, thresolds = roc_curve(self.actual_preds, self.test_inputs)

        return auc(fpr, tpr)      

    def Uncertainty_Context(self, X_test_single, Y_test_single):
        """
        This function calculates the uncertainty of the contextual meaning by
        calculating the votes from all the bags.

        Parameters
        ----------
        X_test_single : pd.series
            one single test example.

        Returns
        -------
        None.

        """

        C = len(self.Response_Variables)  # Number of the classes
        V = np.zeros(C)  # Vote vector
        T = len(self.Bags)  # total number of the bags

        for i in range(T):
            pred = self.Bags[i].predict(X_test_single)
            
            # AUC per bag, combine single tests into one list. Then, predict on the whole set and use that for AUC
            
            
            V[int(pred)] += 1 
        max_value = max(V)
        max_indices = [i for i, value in enumerate(V) if value == max_value]
        choice = random.choice(max_indices)
        self.test_inputs = choice
        self.actual_preds = int(Y_test_single.iloc[0])

        #find majority vote, for 50/50 just choose first value
        #Potentially check weighting of mass vector
        Uncertainty_c = 1 - np.sqrt(np.sum(np.power((V / T - 1 / C), 2))) / np.sqrt(
            (C - 1) / C) #this is part of the mass assignment

        if(Uncertainty_c > 0 and Uncertainty_c < 0.5):
            print(Uncertainty_c)
        # else:
        #     print('0')
        return Uncertainty_c


class ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"
    FULL = "%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d remaining feature sets"

    def __init__(self, total, width=40, fmt=DEFAULT, symbol="=", output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r"(?P<name>%\(.+?\))d", r"\g<name>%dd" % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "remaining": remaining,
        }
        print("\r" + self.fmt % args, file=self.output, end="")

    def done(self):
        self.current = self.total
        self()
        print("", file=self.output)


def feature_set(sensors_to_fuse, st_feat, Sensors, seed_number, feat_set_count=100):
    """
    This function takes a list of name of the sensors to fuse "sensors_to_fuse
    and structure of the reduced feature space also as a list (like [s1,s2,..])
    where "si" is the number of the features of the ith sensor to be used and
    also number of the reduced feature sets to create. Then it returns a matrix
    with each row being a feature set number of columns

    Parameters
    ----------
    sensors_to_fuse: list
        a list of the name of the sensors to fuse. like ['Acc','Gyro','PS','Aud']
    st_feat: list
        the structure of a feature set. like [s1,s2,..] where "si" is
        the number of the features of the ith sensor
    feat_set_count: integer
        the number of randome feature sets to create. It is a number like 100
    Sensors: dict
        the sensors dictionary which is the output of the labeling function

    Returns
    -------
        selected_feats: np.array 2D
            a matrix that each row represents one set of features and the
            values in each rows are the index of the columns of the data to
            be used as features

    """
    # See the PRNG
    random.seed(a=seed_number)

    # Making sure that the length of the "st_feat" is equal to the length of the
    # "sensors_to_fuse
    assert len(st_feat) == len(sensors_to_fuse)

    # Creating and initializing the "selected_feats" to 0
    selected_feats = np.zeros([feat_set_count, sum(st_feat)])

    # A for loop to create the desired number of the random reduced feature sets
    for j in range(feat_set_count):
        # "col" stores the index of the randomly generated columns (features)
        col = []
        for i in range(len(st_feat)):
            # "aux" stores the index of the generated columns of one sensor
            aux = random.sample(Sensors[sensors_to_fuse[i]], st_feat[i])
            aux.sort()
            col.extend(aux)
        selected_feats[j, :] = col
    return selected_feats


def BPA_builder(labels):
    """This function creates the Basic Probability Assignment (BPA) matrix given
    a list of labels (classes)

    Input:
        labels: a list of the target labels to be used as our propositins.
                In other words it is the Frame of discernment (FOD).
                like: ['Walking','Lying_down']
    Ouput:
        mass: a dataframe that has all the subsets of the the FOD. each row
              respresent a sebset with a binary vector like:

                  Lying_down     Sleeping    mass
                  0              0           0
                  0              1           0
                  1              0           0
                  1              1           0

              ex. the last row represents {'Lying_down','Sleeping'} and m is
              the corresponding basic beleif assignment or mass function.
              also all masses are initialized with 0

    """

    mass = pd.DataFrame(columns=labels)
    perms = list(itertools.product([0, 1], repeat=len(labels)))
    for i in range(len(perms)):
        mass.loc[i] = perms[i]
    mass["m"] = 0
    return mass


def pair_resp(BPA):
    """This function takes in the mss function (dataframe in fact) and returns
    two lists. Each element in each of the lists is itself a list of labels
    representing one subset of the FOD or one member of the power set. Note
    that perms_set_1[i] and perms_set_2[i] are complementary subsets.

    Input:
        BPA: the mass dataframe. output of the BPA_builder()

    Output:
        perms_set_1: a list of lists
        perms_set_2: the same

    ex. target labels = 'Lying_down','Sleeping','Walking'
        perms_set_1[2]=['Lying_down','Sleeping']
        perms_set_2[2]=['Walking']

    """

    perms_set_1 = []
    perms_set_2 = []

    # Here we don't need the last column of the mass which is the valuses of mass
    # function. So we get rid of them so that they won't interfere later
    BPA = BPA.iloc[:, :-1]

    # It is enough to go up to the half of the rows of the mass to generate the
    # two classes of permustations. The rest are just the complementary to the
    # first half. In fact if the powerset has 2**n elements, we have 2**(n-1)
    # complementary pairs
    for i in range(BPA.shape[0] // 2):
        assert (
            sum(BPA.loc[i, :]) + sum(BPA.loc[BPA.shape[0] - i - 1, :])
        ) == BPA.shape[1]
        perms_set_1.append([])
        perms_set_2.append([])

        for j in range(BPA.shape[1]):
            if BPA.iloc[i, j] == 1:
                perms_set_1[i].append(BPA.columns[j])
            else:
                perms_set_2[i].append(BPA.columns[j])

    perms_set_1.remove(perms_set_1[0])
    perms_set_2.remove(perms_set_2[0])

    return (perms_set_1, perms_set_2)


def Uncertainty_Bias(response_variables):
    """
    This function calculates the "uncertainty of the biased model" based on the
    frequency of the classes.

    Input:
        response_variables[list of pd.Dataframe]: each element of the list is
        itself a dataframe of one class (response variable).

    Output:
        U[float]: uncertainty of the biased model
    """

    C = len(response_variables)  # number of the classes (response variables)
    I = np.zeros(C)  # an array of the frequencies of the classes

    S = 0  # total samples in the dataset
    for i in range(C):
        I[i] = int(np.sum(response_variables[i] == 1))
        S += I[i]

    U = np.sqrt(np.sum(np.power((I / S - 1 / C), 2))) / np.sqrt((C - 1) / C)
    return U


def Model_Selector(uncertainty_m, models_per_rp, num_rp, fs_axis):
    """This function takes the total uncertainty matrix and chooses the least
    uncertain models for every response permutation.

    Input:
        uncertainty_m[np.array 2d]: Matrix of the total uncertainty for all models.
        models_per_rp[int]: Number of the models to select for each response permutation
        num_rp[int]: Number of the response variables (Also the length of the
              uncertainty matrix along one of the dimensions)
        fs_axis[int]: The axis of the uncertainty matrix along which feature sets
            are laid

    Output:
        selected_models_idx[np.array 2d]: A 2d array that holds the feature set
            indices for each response permutation
    """

    selected_models_idx = np.zeros([num_rp, models_per_rp])

    index_m = np.argsort(uncertainty_m, axis=fs_axis)

    for i in range(models_per_rp):
        selected_models_idx[:, i] = np.argwhere(index_m == i)[:, 1]
    
    return selected_models_idx


def Fuse_and_Predict(
    selected_models_idx, Models, FOD, num_classes, num_rp, models_per_rp
):
    assert len(selected_models_idx) == num_rp
    y_pred = np.zeros([1, num_classes])

    fs_idx = int(selected_models_idx[0][0])
    combined_mass = Models[0][fs_idx].mass
    for i in range(num_rp):
        for j in range(models_per_rp):
            if i == 0 and j == 0:
                continue
            fs_idx = int(selected_models_idx[i][j])
            combined_mass = combined_mass & Models[i][fs_idx].mass

    for i in range(num_classes):
        y_pred[0, i] = combined_mass[{FOD[i]}]

    y_pred_aux = np.zeros_like(y_pred)
    y_pred_aux[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
    y_pred = y_pred_aux

    # return y_pred, combined_mass
    return y_pred


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
        y1 = y1.loc[indices, :]
        y2 = y2.loc[indices, :]
        X = X.loc[indices, :]

    y = y1

    xy = pd.concat([X, y], axis=1)
    xy = xy.dropna()

    y = xy.iloc[:, -1]
    X = xy.iloc[:, 0:-1]

    return (X, y, y1, y2)


# TODO: Figure out why this function was commented out originally
#  I think it was originally part of the main.py code, hence it relies on some
#  of the variables found there
# TODO: What does this do that train_NAPS_Models() doesn't?
def NAPS_Models_Trainer(
    rp_number,
    fs_range,
    train_dataset,
    feature_sets,
    Response_Perm_1,
    Response_Perm_2,
    bagging_ratio,
    num_bags,
    seed_number,
    impute=True,
):
    """
    This function trains NAPS models.

    Parameters
    ----------
    rp_number : int
        integer assigned to response permutation
    fs_range : range or list
        a list of int from 0 to m (m being the total number of the feature sets).
    train_dataset : pd.dataframe
        dataframe of the training features.
    feature_sets : np.array 2D
        matrix of the selected features.
    Response_Perm_1 : list
        lsit of the response permutations.
    Response_Perm_2 : list
        complementary list of the Response_Perm_1.
    impute : bool, optional
        Whether to impute or discard missing features. The default is True.

    Returns
    -------
    None.

    """
    naps_model = []

    # print("\nResponse Permutation {}".format(rp_number + 1))

    for j_fs in fs_range:
        naps_model.append([])

        # find X and y
        X_train, y_train, y1, y2 = Xy(
            train_dataset,
            feature_sets,
            j_fs,
            Response_Perm_1,
            Response_Perm_2,
            rp_number,
            impute=True,
        )

        # NOTE: What does this do?
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        smt = SMOTE()
        X_train_tmp, y_train_tmp = smt.fit_resample(X_train, y_train)

        X_train = pd.DataFrame(X_train_tmp, columns=X_train.columns)
        y_train = pd.DataFrame(y_train_tmp, columns=["Merged_label"])

        # if j_fs == 0:
        #     U2 = Uncertainty_Bias([y1, y2])

        # NOTE: What is used in train_NAPS_Models()
        U2 = Uncertainty_Bias([y1, y2])

        # create a model and train it
        naps_model[j_fs] = DS_Model(
            Response_Perm_1[rp_number],
            Response_Perm_2[rp_number],
            X_train,
            y_train,
            j_fs,
        )
        naps_model[j_fs].Bags_Trainer(X_train, y_train, bagging_ratio, num_bags, seed_number)
        naps_model[j_fs].Uncertainty_B = U2

    return rp_number, naps_model
