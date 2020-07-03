import imageio
import numpy as np
import os

# Change according to your local path/path on Cartesius
# If you work locally
in_path_local = "../../Data/Fold-1/images/fold1/images.npy"
out_path_local = "../../PanNuke Inference Subset"

# For the current Cartesius setup
outh_path_main = "/home/ccurs011/PanNuke Inference/"
in_path_1 = "/projects/0/ismi2018/PanNuke/Fold-1/images/fold1/images.npy"
out_path_1 = outh_path_main + "fold1"
in_path_2 = "/projects/0/ismi2018/PanNuke/Fold-2/images/fold2/images.npy"
out_path_2 = outh_path_main + "fold2"
in_path_3 = "/projects/0/ismi2018/PanNuke/Fold-3/images/fold3/images.npy"
out_path_3 = outh_path_main + "fold3"

# For stain normalized images
outh_path_main = "/home/ccurs011/PanNuke Inference SN/"
in_path_1 =  "/home/ccurs011/HoverNet/PanT/fold1/"
out_path_1 = outh_path_main + "fold1"
in_path_2 = "/home/ccurs011/HoverNet/PanT/fold2/"
out_path_2 = outh_path_main + "fold2"
in_path_3 = "/home/ccurs011/HoverNet/PanT/fold3/"
out_path_3 = outh_path_main + "fold3"

# Converts a sample of a specified length from input folder to .png
# If length=None converts all the images
# Assumes that all images in one images.npy
def convert_sample(in_path, out_path, length=None):
    # We would get fold1, fold2 or fold3 this way
    prefix = os.path.split(os.path.split(in_path)[0])[1]
    images = np.load(in_path)
    images_num = length if length is not None else images.shape[0]
    for i in range(images_num):
        print("Converting image {}".format(i+1))
        img = images[i]
        img_uint = np.array(img, np.uint8)
        imageio.imsave("{}/{}_{}.png".format(out_path, prefix, i+1), img_uint)

def convert_whole_dataset():
    try:
        os.mkdir(outh_path_main)
    except FileExistsError:
        pass

    out_dirs = [out_path_1, out_path_2, out_path_3]

    for out_dir in out_dirs:
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            pass

    convert_sample(in_path_1, out_path_1)
    convert_sample(in_path_2, out_path_2)
    convert_sample(in_path_3, out_path_3)


# Converts a sample of a specified length from input folder to .png
# If length=None converts all the images
# Assumes that all images in separate images.npy
def convert_sample_stain_norm(in_path, out_path, length=None):
    for file in os.listdir(in_path):
        if file.endswith(".npy"):
            file_name = os.path.splitext(file)[0]
            print("Converting image {}".format(file_name))
            img = np.load(os.path.join(in_path, file))
            img = img[:, :, :3]
            print(img.shape)
            img_uint = np.array(img, np.uint8)
            png_file_name = "{}.png".format(file_name)
            save_path = os.path.join(out_path, png_file_name)
            imageio.imsave(save_path, img_uint)


def convert_whole_dataset_stain_norm():
    try:
        os.mkdir(outh_path_main)
    except FileExistsError:
        pass

    out_dirs = [out_path_1, out_path_2, out_path_3]

    for out_dir in out_dirs:
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            pass

    convert_sample_stain_norm(in_path_1, out_path_1)
    convert_sample_stain_norm(in_path_2, out_path_2)
    convert_sample_stain_norm(in_path_3, out_path_3)

if __name__ == '__main__':
    convert_whole_dataset_stain_norm()
