"""
NAPS Fusion configuration file
"""
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
feature_range = range(1, 225)  # range of the column number of all features
num_prc = 7  # number of processors to split the job during parallelization
parallelize = True
random_seed_number = 42
