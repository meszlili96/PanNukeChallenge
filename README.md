# PanNukeChallenge

Based on the implementation of [Quoc Dang Vu](https://github.com/vqdang) and [Simon Graham](https://github.com/simongraham) available at [this link](https://github.com/vqdang/hover_net).

## Steps to Reproduce Results
1. Access PanNuke data and [pretrained weights](https://drive.google.com/file/d/187C9pGjlVmlqz-PlKW1K8AYfxDONrB0n/view)
2. Add the location of the weights in `hover.py` and `other.py`.
3. Transform the PanNuke data with `transform_pannuke.py` to the correct format of `(img_size,img_size,5)`, where the indices 0:3 are the RGB codes, 3 is the instance and 4 is the class type.
4. An optional step is stain normalisation with `normalizeStaining.py`. This is added to `transform_pannuke.py`, but can be changed in the settings of the `transform` function.
5. Micro-Net and Hover-Net need to be trained separately with certain parameters:
```
python train.py --gpu='gpu_ids' --train_dir='train_dirs' --valid_dir='valid_dir'
```
For example, if we are using GPU number 0 and 1, the training directories are `fold1` and `fold2`, and the validation direcory is `fold3` the command is:
python train.py --gpu='0,1' --train_dir='fold1,fold2' --valid_dir='fold3'
6. Inference can be run by `python infer.py --gpu='gpu_id'`. Note that only one GPU is supported.
7. Final instance segmentation can be obtained by running `python process.py`.
