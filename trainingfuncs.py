import random
import numpy as np
from tensorflow.keras import layers, Sequential, datasets, models, losses, Model, Input
import tensorflow as tf

def slice_matrix(matrix, n_slices, slice_shape, double_index=True, permute=True):
    '''
    slice a matrix vertically and/or horizontally into slices
    n_slices: tuple, (vertical_slices, horizontal_slices)
    slice_shape: tuple, (h, w)
    '''
    if (n_slices[0]==1) or (n_slices[1]==1):
        double_index=False
    total_slices = np.product(n_slices)
    
    h, w = matrix.shape
    h_desired, w_desired = slice_shape
    h_max = h//n_slices[0]
    w_max = w//n_slices[1]
    assert (h_max>=h_desired) and (w_max>=w_desired)
    slice_list = []
    index_list = []
    
    h_margin = h_max-h_desired
    w_margin = w_max-w_desired
    
    for row in range(n_slices[0]):
        for col in range(n_slices[1]):
            row_i = row*h_max+random.randint(0, h_margin)
            col_i = col*w_max+random.randint(0, w_margin)
            slice_list.append(matrix[row_i:row_i+h_desired, col_i:col_i+w_desired])
            if double_index:
                index_list.append([row, col])
            else:
                index_list.append(row*n_slices[1]+col)
    assert (len(slice_list)==total_slices) and (len(index_list)==total_slices)
    
    if permute:
        slice_list, index_list = permute_lists(slice_list, index_list)
#         shuffled = list(range(total_slices))
#         random.shuffle(shuffled)
#         shuffled_slice = [slice_list[i] for i in shuffled]
#         shuffled_index = [index_list[i] for i in shuffled]
#         slice_list = shuffled_slice
#         index_list = shuffled_index
    slice_list_3d = [np.stack([s, s, s], axis = 2) for s in slice_list]
    slice_list = slice_list_3d
    return np.array(slice_list), np.array(index_list).astype(float)

def alexnet(input_shape, trainable=True):
    model = Sequential()
    model.add(layers.experimental.preprocessing.Resizing(128, 128, interpolation="bilinear", input_shape=input_shape))
    model.add(layers.Conv2D(96, 11, strides=2, padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(3, strides=2, padding='valid'))
    model.add(layers.Conv2D(256, 5, padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(3, strides=2, padding='valid'))
    model.add(layers.Conv2D(384, 3, padding='same', activation='relu'))
    model.add(layers.Conv2D(384, 3, padding='same', activation='relu'))
    model.add(layers.Conv2D(256, 3, padding='same', activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    if not trainable:
        model.trainable=False
    model.summary()
    return model

def self_VGG(weights=None, trainable=True):
    if weights!='imagenet' and weights is not None:
        raise ValueError('Non-imagenet weights not supported yet.')
        
    img_model = tf.keras.applications.VGG19(include_top=False, weights=weights, 
                                        input_shape=(224, 224, 3), pooling='avg')
    
    if not trainable:
        img_model.trainable=False
        
    return img_model

def permute_lists(iterable, *args, return_index=False):
    n_iterable = len(list(iterable))
    
    shuffled = list(range(n_iterable))
    random.shuffle(shuffled)
    iterable_shuffled = []
    iterable_shuffled.append([iterable[i] for i in shuffled])
    for arg in args:
        iterable_shuffled.append([arg[i] for i in shuffled])
    if return_index:
        iterable_shuffled.append(shuffled)
    return iterable_shuffled