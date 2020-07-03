import numpy as np
import os
import re

transformed_dir = "/home/ccurs011/HoverNet/PanT/"
save_dir = "/home/ccurs011/Cleaned data"

images_1 = '/projects/0/ismi2018/PanNuke/Fold-1/images/fold1/images.npy'
masks_1 = '/projects/0/ismi2018/PanNuke/Fold-1/masks/fold1/masks.npy'
types_1 = '/projects/0/ismi2018/PanNuke/Fold-1/images/fold1/types.npy'
images_2 = '/projects/0/ismi2018/PanNuke/Fold-2/images/fold2/images.npy'
masks_2 = '/projects/0/ismi2018/PanNuke/Fold-2/masks/fold2/masks.npy'
types_2 = '/projects/0/ismi2018/PanNuke/Fold-2/images/fold2/types.npy'
images_3 = '/projects/0/ismi2018/PanNuke/Fold-3/images/fold3/images.npy'
masks_3 = '/projects/0/ismi2018/PanNuke/Fold-3/masks/fold3/masks.npy'
types_3 = '/projects/0/ismi2018/PanNuke/Fold-3/images/fold3/types.npy'


def remove_lost_images(images, masks, types, lost_ids, fold):
    cleaned_images = np.delete(images, lost_ids, 0)
    cleaned_masks = np.delete(masks, lost_ids, 0)
    cleaned_types = np.delete(types, lost_ids, 0)

    full_save_dir = os.path.join(save_dir, fold)
    try:
        os.mkdir(full_save_dir)
    except FileExistsError:
        pass

    images_name = os.path.join(full_save_dir, "images")
    np.save(images_name, cleaned_images)
    print("Cleaned images num {}".format(len(cleaned_images)))
    masks_name = os.path.join(full_save_dir, "masks")
    np.save(masks_name, cleaned_masks)
    print("Cleaned masks num {}".format(len(cleaned_masks)))
    types_name = os.path.join(full_save_dir, "types")
    np.save(types_name, cleaned_types)
    print("Cleaned types num {}".format(len(cleaned_types)))


def lost_ids(folder, expected_ids, fold):
    files = []
    for file in os.listdir(folder):
        if file.endswith('.npy'):
            files.append(file)

    sort_re = re.compile(r"{}_(\d+).npy".format(fold))
    ids = [int(sort_re.match(x).groups()[0]) - 1 for x in files]
    ids.sort()
    missed_ids = [x for x in expected_ids if x not in ids]
    return missed_ids


def process(transformed_dir, images_path, mask_path, types_path, fold):
    try:
        os.mkdir(transformed_dir)
    except FileExistsError:
        pass

    images = np.load(images_path)
    masks = np.load(mask_path)
    types = np.load(types_path)

    expected_ids = range(len(images))
    fold_dir = os.path.join(transformed_dir, fold)
    print("Processing dir {}".format(fold_dir))
    lost = lost_ids(fold_dir, expected_ids, fold)
    print("In fold {} the following ids are lost {}".format(fold, lost))

    remove_lost_images(images, masks, types, lost, fold)


process(transformed_dir, images_1, masks_1, types_1, "fold1")
process(transformed_dir, images_2, masks_2, types_2, "fold2")
process(transformed_dir, images_3, masks_3, types_3, "fold3")
