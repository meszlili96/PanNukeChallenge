# PanNukeChallenge

Based on the implementation of [Quoc Dang Vu](https://github.com/vqdang) and [Simon Graham](https://github.com/simongraham) available at [this link](https://github.com/vqdang/hover_net).

## Steps to Reproduce Results
0. Install the requirements (it is probably easier to create a separate environment with conda):
```
pip install -r requirements.txt
```
1. Access PanNuke data and [pretrained weights](https://drive.google.com/file/d/187C9pGjlVmlqz-PlKW1K8AYfxDONrB0n/view).
2. Add the location of the weights in `hover.py` and `other.py`.
3. Transform the PanNuke data with `transform_pannuke.py` to the correct format of `(img_size,img_size,5)`, where the indices 0:3 are the RGB codes, 3 is the instance and 4 is the class type. The output directory (`out_dir`) has to be set manually.
4. An optional step is stain normalisation with `normalizeStaining.py`. This is added to `transform_pannuke.py`, but can be changed in the settings of the `transform` function.
5. In `config.py` the model type and paths need to be set depending on if we are training Micro-Net or Hover-Net at the moment. 
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
6. Training can be started with passing the GPU ids as arguments, e.g. with 0 and 1.
```
python train.py --gpu='0,1'
```
7. Overall, the config and training needs to be edited/started 6 times (3 folds times 2 models). The training can be done parallel, with creating duplicates of the config and the training files. 
8. Inference can be run by `python infer.py --gpu='gpu_id'`. Note that only one GPU is supported. As in the previous step, this needs to be done 6 times, using the right configuration. In `config.py` the path to the model checkpoint, the inference images, and the output directory needs to be given:
```
self.inf_model_path  = self.save_dir + '/model-19640.index'
self.inf_data_dir = '/inference/'
self.inf_output_dir = '/output/'
```
9. Final instance segmentation can be obtained by running `python process.py`.
