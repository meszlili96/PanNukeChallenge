# PanNuke Challenge

* Micro-Net and Hover-Net are based on the implementation of [Quoc Dang Vu](https://github.com/vqdang) and [Simon Graham](https://github.com/simongraham) available at [this link](https://github.com/vqdang/hover_net).
* The evaluation is taken from [TIA-Lab/PanNuke-metrics](https://github.com/TIA-Lab/PanNuke-metrics).

![](pipeline.png)

## Steps to Reproduce Results
### Prerequisites
1. Install the requirements (it is probably easier to create a separate environment with conda):
```
pip install -r requirements.txt
```
2. Access PanNuke data and [pretrained weights](https://drive.google.com/file/d/187C9pGjlVmlqz-PlKW1K8AYfxDONrB0n/view).
3. Add the location of the weights in `src/opt/hover.py` and `src/opt/other.py`.

### Preprocessing
4. Transform the PanNuke data with `src/transform_pannuke.py` to the correct format of `(img_size,img_size,5)`, where the indices 0:3 are the RGB codes, 3 is the instance and 4 is the class type. The output directory (`out_dir`) has to be set manually.
5. An optional step is stain normalisation with `src/normalizeStaining.py`. This is added to `src/transform_pannuke.py`, but can be changed in the settings of the `transform` function.

### Training
6. In `src/config.py` the model type and paths need to be set depending on if we are training Micro-Net or Hover-Net at the moment. 
* Hover-Net:
```
mode = 'hover'
self.model_type = 'np_hv'
```
* Micro-Net:
```
mode = 'other'
self.model_type = 'micronet'
```
* For the logging different paths need to be given, e.g. `self.log_path = '/hovernet_logs/'` and `self.log_path = '/micronet_logs/'`.
* Since PanNuke contained 3 folds, for both models we used cross validation, with 2 folds for training, 1 for validation:
```
self.train_dir = ['/fold1/','/fold2/']
self.valid_dir = ['/fold3/']
```
7. Training can be started with passing the GPU ids as arguments, e.g. with 0 and 1.
```
python train.py --gpu='0,1'
```
8. Overall, the config and training needs to be edited/started 6 times (3 folds times 2 models). The training can be done parallel, with creating duplicates of the config and the training files.

### Inference
9. Inference can be run by `python src/infer.py --gpu='gpu_id'`. Note that only one GPU is supported. As in the previous step, this needs to be done 6 times, using the right configuration. In `src/config.py` the path to the model checkpoint, the inference images, and the output directory needs to be given:
```
self.inf_model_path  = self.save_dir + '/model-19640.index'
self.inf_data_dir = '/inference/'
self.inf_output_dir = '/output/'
```
10. Final instance segmentation can be obtained by running `python src/process.py`.

### Evaluation
11. For the segmentation metrics `src/metrics/transform_for_metrics.py` needs to be run to convert the predictions back to the PanNuke format with 6 channels. The path to the resulting .mat files and the output .npy file is necessary. This will look something like the following:
```
path_to_folder = '/output/v3.0/micronet/_proc' 
output = 'masks.npy'
```
12. Run `src/metrics/remove_lostids_from_gt_fold3.py` and `src/metrics/remove_lostids_from_gt_fold1.py` to remove the lost files from the ground truth. 
13. Now, `python src/metrics/run.py --true_path=<n> --pred_path=<n> --save_path=<n>y` can be used to obtain the panoptic quality scores.
13. For the detection/classification metrics, run `python convert_to_centroids.py` to extract the centroids of the ground truth.
14. Now, `src/compute_stats.py` in the repository vqdang/hover_net can be run with ground truth being the extracted centroids and the predictions being post-processed predictions (.mat files).
