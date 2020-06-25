import numpy as np
import tensorflow as tf
from .misc import * 

micronet = {
    'train_input_shape' : [256, 256],
    'train_mask_shape'  : [256, 256],
    'infer_input_shape' : [256, 256],
    'infer_mask_shape'  : [256, 256], 

    'training_phase'    : [
        {
            'nr_epochs': 250,
            'manual_parameters' : {
                'learning_rate': (1.0e-4, [('125', 1.0e-5)]),
                'aux_loss_dw'  : (1.0, 
                        [(str(epoch), 1.0 / epoch) for epoch in range(2, 251)]
                    ),
            },
            'train_batch_size'  : 4,
            'infer_batch_size'  : 8,

            'model_flags' : {
            }
        }
    ],

    'optimizer'         : tf.train.AdamOptimizer,

    'inf_batch_size'  : 8,
    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
}
