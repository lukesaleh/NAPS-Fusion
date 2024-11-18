"""
NAPS Fusion configuration file
"""
import numpy as np

# If are using UCSD Dataset type True into Using_UCSD and if you're using any other data set type False

Using_UCSD = False # Type True or False here

if Using_UCSD == True:

    results_dir = "../results/naps.baseline_original"
    data_dir = "../data/NAPS-Fusion/datasets/"
    cvdir = "../data/NAPS-Fusion/cv5Folds/cv_5_folds/"

# There are also "Mag", "AP", "LF", "Compass" sensors
# NOTE: Should it be WAcc instead of W_acc? Nope, code gives sensor a different
# name
    sensors_to_fuse = ["Acc", "Gyro", "W_acc", "Aud"]
    FOD = [
    "label:LYING_DOWN",
    "label:SITTING",
    "label:OR_standing",
    "label:FIX_walking",
    ]
# Feature set structure
# NOTE: Must match the length of sensors_to_fuse
# feature_sets_st = [3, 3, 3, 3] # original # features in reduced feature set
    feature_sets_st = [3] * len(sensors_to_fuse)
    feature_sets_count = 10
    bagging_R = 0.6  # bagging ratio 
    num_bags = 4
    models_per_rp = 2  # number of models to select for the fusion per response permutation
    feature_range = range(1,225)
    num_prc = 7  # number of processors to split the job during parallelization
    parallelize = True
    random_seed_number = 42

else:
    # Enter the directory you want your results to go
    results_dir = "" 
    # Enter the directory your data set is coming from
    data_dir = ""
     

    # Enter the features and class labels you'd wish to use respectively
    sensors_to_fuse = [] 

    # No more than 6 Labels
    FOD = [
    "label: CLASS_PRED1_HERE ",
    "label: CLASS_PRED2_HERE ",
    "label: CLASS_PRED3_HERE ",
    "label: CLASS_PRED4_HERE ",
    ]

    def Set_Act_Sens_NEW():

     """This function defines two dictionaries for activities and sensors. Each
     dictionaray holds the the range of columns for the specified sensor or
     activity.

     Input:
     Output:
        Activities[dict]: a dictionary of the TARGET CLASS LABELS and their corresponding
                        column number
        Sensors[dict]: a dictionary of the FEATURE LABELS and their corresponding range
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



# Feature set structure
# NOTE: Must match the length of sensors_to_fuse
# feature_sets_st = [3, 3, 3, 3] # original # features in reduced feature set
    feature_sets_st = [3] * len(sensors_to_fuse)
    feature_sets_count = 10
    bagging_R = 0.6  # bagging ratio 
    # Enter the number of bags you would like to use
    num_bags = 4
    models_per_rp = 2  # number of models to select for the fusion per response permutation
    # Enter the feature range for you data set
    feature_range = range(1,225)
    num_prc = 7  # number of processors to split the job during parallelization
    parallelize = True
    random_seed_number = 42


