#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:10:55 2020

@author: carotenuto
"""

import numpy as np

masks = np.load("/projects/0/ismi2018/PanNuke/Fold-3/masks/fold3/masks.npy")
types = np.load("/projects/0/ismi2018/PanNuke/Fold-3/images/fold3/types.npy")

lost_ids = [762, 1566, 1569, 1777, 1804, 1834, 1837, 1838, 1839, 1840, 1841, 1854, 1855, 1856, 1860, 1861, 1862, 1897, 2023]

for j, lost_id in enumerate(lost_ids):
    print(j)
    print(lost_id)
    masks = np.delete(masks, (lost_id-j), axis=0)
    types = np.delete(types, (lost_id-j), axis=0)

print(masks.shape)
print(types.shape)
#np.save("/scratch-local/ccurs011/gt_fold3_TIA-metrics/masks.npy", masks)
#np.save("/scratch-local/ccurs011/gt_fold3_TIA-metrics/types.npy", types)
np.save("/home/ccurs011/TIA-Lab-Pannuke-metrics/fold3/gt/masks.npy", masks)
np.save("/home/ccurs011/TIA-Lab-Pannuke-metrics/fold3/gt/types.npy", types)
