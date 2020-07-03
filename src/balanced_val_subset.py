import os
import numpy as np
import shutil

"""The script to extract balanced subset of validation data for ensemble weights training
"""

# Paths to types by fold
fold_1 = '/home/ccurs011/types/fold1/types.npy'
fold_2 = '/home/ccurs011/types/fold2/types.npy'
fold_3 = '/home/ccurs011/types/fold3/types.npy'

# The images ids missed during stain normalization for each fold
missed_ids1 = [1723, 1939, 1940, 1946, 1969, 1973, 1977, 1978, 1979, 2009, 2020, 2022]
missed_ids2 = [1471, 1472, 1543, 1607, 1651, 1653, 1661, 1662, 1685, 1686, 1687, 1688, 1692, 1694, 1702, 1889]
missed_ids3 = [762, 1566, 1569, 1777, 1804, 1834, 1837, 1838, 1839, 1840, 1841, 1854, 1855, 1856, 1860, 1861, 1862, 1897, 2023]

# The paths to validation fold for each split
dataset_path_split1 = "/home/ccurs011/PanNuke Inference SN/fold2"
dataset_path_split2 = "/home/ccurs011/PanNuke Inference SN/fold1"
dataset_path_split3 = "/home/ccurs011/PanNuke Inference SN/fold2"

# The paths to save the subset
subset_path_split1 = "/home/ccurs011/PanValidation Subset/split1"
subset_path_split2 = "/home/ccurs011/PanValidation Subset/split2"
subset_path_split3 = "/home/ccurs011/PanValidation Subset/split3"

# Substitute the type with None for lost indices
types_1 = np.load(fold_1)
types_1[missed_ids1] = "None"
types_2 = np.load(fold_2)
types_2[missed_ids2] = "None"
types_3 = np.load(fold_3)
types_3[missed_ids3] = "None"

# Create dictionaries of indices by type
unique_types = np.unique(types_1)

fold1_types_idxs = {}
fold2_types_idxs = {}
fold3_types_idxs = {}

for type in unique_types:
    if type != "None":
        indices1 = [i for i, x in enumerate(types_1) if x == type]
        fold1_types_idxs[type] = indices1

        indices2 = [i for i, x in enumerate(types_2) if x == type]
        fold2_types_idxs[type] = indices2

        indices3 = [i for i, x in enumerate(types_3) if x == type]
        fold3_types_idxs[type] = indices3


# Generate balanced data sample with type_num images of each tissue type
def balanced_data_subset(ids_by_type, type_num=10):
    indexes = []
    for _, type_ids in ids_by_type.items():
        sample = np.array(type_ids) if len(type_ids) < type_num else np.random.choice(type_ids, type_num, replace=False)
        indexes.extend(sample.tolist())

    return indexes


def copy_files(source_dir, dest_dir, ids):
    for id in ids:
        fold = os.path.basename(source_dir)
        filename = "{}_{}.png".format(fold, id+1)
        shutil.copy("{}/{}".format(source_dir, filename), dest_dir)
        print("{} copied".format(filename))


sample_split1 = balanced_data_subset(fold2_types_idxs)
print("Split1")
print(sample_split1)
copy_files(dataset_path_split1, subset_path_split1, sample_split1)

sample_split2 = balanced_data_subset(fold1_types_idxs)
print("Split2")
print(sample_split2)
copy_files(dataset_path_split2, subset_path_split2, sample_split2)

sample_split3 = balanced_data_subset(fold2_types_idxs)
print("Split3")
print(sample_split3)
copy_files(dataset_path_split3, subset_path_split3, sample_split3)
