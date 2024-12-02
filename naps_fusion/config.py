"""
NAPS Fusion configuration file
"""
import numpy as np

# If are using UCSD Dataset type True into Using_UCSD and if you're using any other data set type False

import pandas as pd

def convert_ground_truth_to_binary(input_file, output_file):
    """
    Reads a CSV file, extracts features and ground truth labels, and converts the labels into binary columns.
    
    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to save the output CSV file with binary class columns.
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Assume the last column is the ground truth
    ground_truth = df.iloc[:, -1]
    
    # Get unique class labels
    unique_classes = sorted(ground_truth.unique())
    
    # Create binary columns for each class
    for cls in unique_classes:
        df[f'Class_{cls}'] = (ground_truth == cls).astype(int)
    
    # Drop the original ground truth column
    df = df.drop(columns=df.columns[-2])  # Drop the original class column (before binary transformation)
    
    # Save the resulting DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Transformed CSV file saved to {output_file}")


def get_non_class_columns(csv_file):
    """
    Loads a CSV file and returns the names of all columns except the class columns.
    
    Parameters:
    - csv_file (str): Path to the input CSV file.
    
    Returns:
    - list: A list of column names excluding class columns.
    """
    # Load the DataFrame from the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter out columns whose names start with "Class_"
    non_class_columns = [col for col in df.columns if not col.startswith('Class_')]
    
    return non_class_columns



Using_UCSD = True # Type True or False here

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
    parallelize = False
    random_seed_number = 42

else:
    # Enter the directory you want your results to go
    results_dir = "" 
    # Enter the directory your data set is coming from and where reshaped data should go
    data_dir = ""
    new_data_dir = ""

    #function to convert truth labels for tabular data
    convert_ground_truth_to_binary(data_dir, new_data_dir)

    data_dir = new_data_dir

    #Enter the column in the data which corresponds to target labels
    label_column = 0 

    # Enter the features and class labels you'd wish to use respectively. Leave empty to fuse all sensors
    sensors_to_fuse = [] 

    if sensors_to_fuse is []:
        sensors_to_fuse = get_non_class_columns(data_dir)
        print('Dropping no sensors')

    # No more than 6 Labels in FOD
    FOD = [
    "CLASS_PRED1_HERE ",
    "CLASS_PRED2_HERE ",
    "CLASS_PRED3_HERE ",
    "CLASS_PRED4_HERE ",
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
    
    #Group features by "sensor"
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
    parallelize = False
    random_seed_number = 42


