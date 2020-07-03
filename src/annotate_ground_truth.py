import cv2
import os
import glob
import numpy as np
import itertools

"""The script to create the images with ground truth annotation highlighted with overlays of different colors
"""

### visualize_instances, class_colour and bounding_box are taken from https://github.com/simongraham/hovernet_inference
####
def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def class_colour(class_value):
    """
    Generate RGB colour for overlay based on class id
    Args:
        class_value: integer denoting the class of object
    """
    if class_value == 0:
        return 0, 0, 0  # black (background)
    if class_value == 1:
        return 255, 0, 0  # red
    elif class_value == 2:
        return 0, 255, 0  # green
    elif class_value == 3:
        return 0, 0, 255  # blue
    elif class_value == 4:
        return 255, 255, 0  # yellow
    elif class_value == 5:
        return 255, 165, 0  # orange
    elif class_value == 6:
        return 0, 255, 255  # cyan
    else:
        raise Exception(
            'Currently, overlay_segmentation_results() only supports up to 6 classes.')


####
def visualize_instances(input_image, predict_instance, predict_type=None, line_thickness=2):
    """
    Overlays segmentation results on image as contours
    Args:
        input_image: input image
        predict_instance: instance mask with unique value for every object
        predict_type: type mask with unique value for every class
        line_thickness: line thickness of contours
    Returns:
        overlay: output image with segmentation overlay as contours
    """

    overlay = np.copy((input_image).astype(np.uint8))

    if predict_type is not None:
        type_list = list(np.unique(predict_type))  # get list of types
        type_list.remove(0)  # remove background
    else:
        type_list = [4]  # yellow

    for iter_type in type_list:
        if predict_type is not None:
            label_map = (predict_type == iter_type) * predict_instance
        else:
            label_map = predict_instance
        instances_list = list(np.unique(label_map))  # get list of instances
        instances_list.remove(0)  # remove background
        contours = []
        for inst_id in instances_list:
            instance_map = np.array(
                predict_instance == inst_id, np.uint8)  # get single object
            y1, y2, x1, x2 = bounding_box(instance_map)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= predict_instance.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= predict_instance.shape[0] - 1 else y2
            inst_map_crop = instance_map[y1:y2, x1:x2]
            contours_crop = cv2.findContours(
                inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            index_correction = np.asarray([[[[x1, y1]]]])
            for i in range(len(contours_crop[0])):
                contours.append(
                    list(np.asarray(contours_crop[0][i].astype('int32')) + index_correction))
        contours = list(itertools.chain(*contours))
        cv2.drawContours(overlay, np.asarray(contours), -1,
                         class_colour(iter_type), line_thickness)
    return overlay
####


def covert(npy_dir, output_dir, fold):
    npy_dir_fold = os.path.join(npy_dir, fold)
    output_dir_fold = os.path.join(output_dir, fold)

    try:
        os.mkdir(output_dir_fold)
    except FileExistsError:
        pass

    gt_numpy_list = glob.glob(npy_dir_fold + '/*.npy')
    # Image has HoverNet input format, first 3 channels are RGB image, 4 - instances, 5 - types
    for file in gt_numpy_list:
        filename = os.path.basename(file)
        basename = os.path.splitext(filename)[0]
        numpy_file = np.load(file)
        image = numpy_file[:, :, :3]
        inst_map = numpy_file[:, :, 3]
        type_map = numpy_file[:, :, 4]
        overlaid_output = visualize_instances(image, inst_map, type_map)
        overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
        cv2.imwrite('%s/%s.png' % (output_dir_fold, basename), overlaid_output)


def main():
    npy_dir = "/home/ccurs011/PanNukeSN GT annotated"
    output_dir = "/home/ccurs011/PanNukeSN GT annotated"

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    covert(npy_dir, output_dir, "fold1")
    covert(npy_dir, output_dir, "fold2")
    covert(npy_dir, output_dir, "fold3")


if __name__ == '__main__':
    main()
