import numpy as np
import os
import normalizeStaining as NS

images_1 = np.load('/projects/0/ismi2018/PanNuke/Fold-1/images/fold1/images.npy')
masks_1 = np.load('/projects/0/ismi2018/PanNuke/Fold-1/masks/fold1/masks.npy')
images_2 = np.load('/projects/0/ismi2018/PanNuke/Fold-2/images/fold2/images.npy')
masks_2 = np.load('/projects/0/ismi2018/PanNuke/Fold-2/masks/fold2/masks.npy')
images_3 = np.load('/projects/0/ismi2018/PanNuke/Fold-3/images/fold3/images.npy')
masks_3 = np.load('/projects/0/ismi2018/PanNuke/Fold-3/masks/fold3/masks.npy')
out_dir = "/home/ccurs011/HoverNet/PanT/"

# A helper function to map 2d numpy array
def flat_for(a, f):
    a = a.reshape(-1)
    for i, v in enumerate(a):
        a[i] = f(v)


# A helper function to unique PanNuke instances indexes to [0..N] range where 0 is background
def map_inst(inst):
    seg_indexes = np.unique(inst)
    new_indexes = np.array(range(0, len(seg_indexes)))
    dict = {}
    for seg_index, new_index in zip(seg_indexes, new_indexes):
        dict[seg_index] = new_index

    flat_for(inst, lambda x: dict[x])


# A helper function to transform PanNuke format to HoverNet data format
def transform(images, masks, path, out_dir, norm = False):
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    fold_path = out_dir+path
    try:
        os.mkdir(fold_path)
    except FileExistsError:
        pass

    for i in range(images.shape[0]):
        stop = False
        np_file = np.zeros((256,256,5), dtype='int16')

        # add rgb channels to array
        img_int = np.array(images[i],np.int16)
        for j in range(3):
            np_file[:,:,j] = img_int[:,:,j]
        
        if norm:
            
            #Normailze stain
            try:
                Inorm, H, E = NS.normalizeStaining(np.array(np_file[:,:,:3],np.int16))
                for rgb in range(3):
                    np_file[:,:,rgb] = Inorm[:,:,rgb]

            except:
                print("file %d not normalized" % (i+1))
                stop = True
        

        # convert inst and type format for mask
        msk = masks[i]

        inst = np.zeros((256,256))
        for j in range(5):
            #copy value from new array if value is not equal 0
            inst = np.where(msk[:,:,j] != 0, msk[:,:,j], inst)
        map_inst(inst)

        types = np.zeros((256,256))
        for j in range(5):
            # write type index if mask is not equal 0 and value is still 0
            types = np.where((msk[:,:,j] != 0) & (types == 0), j+1, types)

        # add padded inst and types to array
        np_file[:,:,3] = inst
        np_file[:,:,4] = types
        if stop == False:
            np.save(fold_path +'/'+path+'_%d.npy' % (i+1), np_file)
            print('file %d saved' % (i+1))

# For the correct evaluation we need to keep training, validation and test folds
transform(images_1, masks_1, 'fold1', out_dir=out_dir, norm=True)
transform(images_2, masks_2, 'fold2', out_dir=out_dir, norm=True)
transform(images_3, masks_3, 'fold3', out_dir=out_dir, norm=True)

print("done")
