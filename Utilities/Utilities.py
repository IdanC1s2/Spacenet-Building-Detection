import tensorflow as tf
from tensorflow.keras.callbacks import History
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from functools import reduce
from itertools import chain
import operator
from time import time, sleep
import natsort
import gc
import tensorflow.keras.backend as K


def visualize_history(history):
    # We want to visualize the results. Specifically, the losses and the classification mean_iou of the training and validation data:
    """
    This function is for visualization of the model training process.
    Specifically, we plot 2 plots
    :param history: history object,
        either a standard History class from tf.keras.callbacks, or history dictionary
        which was obtained by 'merge_histories_in_directory()' function or 'load_history_into_dict(...)' function defined below
    :return: A figure containing the plots
    """
    if isinstance(history, History):  # If it is a standard history object
        mean_iou = history.history.get('my_mean_iou')
        val_mean_iou = history.history.get('val_my_mean_iou')
        loss = history.history.get('loss')
        val_loss = history.history.get('val_loss')
    elif isinstance(history, dict):  # If it's a history loaded using: history = merge_histories_in_directory(...) function defined below
        # Or if it was loaded by using: history = load_history_into_dict(...) function defined below
        mean_iou = history.get('my_mean_iou')
        val_mean_iou = history.get('val_my_mean_iou')
        loss = history.get('loss')
        val_loss = history.get('val_loss')
    else:
        raise TypeError
    fig, ax = plt.subplots(1, 2)
    plt.sca(ax[0])
    line1, = plt.plot(mean_iou)
    line2, = plt.plot(val_mean_iou)
    plt.title('Mean IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend([line1, line2], ['Training data', 'Validation data'])
    plt.sca(ax[1])
    line3, = plt.plot(loss)
    line4, = plt.plot(val_loss)
    plt.title('Loss Function')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend([line3, line4], ['Training data', 'Validation data'])

    return fig


class MetaFit:
    """
    MetaFit class is a class we use to hold parameters for our model.fit() function.
    An object of this class will hold in its attributes all of the default parameters used
    in the model.fit function. We can set the parameters as we'd like if we want to change any of them,
    making it easier for us to insert the required parameters to the model.fit() function used inside of our
    custom train_in_parts() function.
    """

    def __init__(self, x=None, y=None, batch_size=None, epochs=1, verbose=1,
                 callbacks=None, validation_split=0., validation_data=None,
                 shuffle=True, class_weight=None, sample_weight=None,
                 initial_epoch=0, steps_per_epoch=None, validation_steps=None,
                 validation_batch_size=None, validation_freq=1, max_queue_size=10,
                 workers=1, use_multiprocessing=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.shuffle = shuffle
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.initial_epoch = initial_epoch
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.validation_batch_size = validation_batch_size
        self.validation_freq = validation_freq
        self.max_queue_size = max_queue_size
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing


def model_fit(model, meta_data):
    """

    :param model: The model we want to train.
    :param meta_data: A MetaFit class object, containing the parameters we want to give the model.fit() function.
    :return: history object of the fitting process
    """

    history = model.fit(meta_data.x, meta_data.y, meta_data.batch_size, meta_data.epochs,
                        meta_data.verbose, meta_data.callbacks, meta_data.validation_split,
                        meta_data.validation_data, meta_data.shuffle, meta_data.class_weight,
                        meta_data.sample_weight, meta_data.initial_epoch, meta_data.steps_per_epoch,
                        meta_data.validation_steps, meta_data.validation_batch_size,
                        meta_data.validation_freq, meta_data.max_queue_size,
                        meta_data.workers, meta_data.use_multiprocessing)
    return history


def train_in_parts(model, meta_data, epochs, epochs_per_iter, cooldown_time=3, dir='.', hist_starting_index=0):
    """
    :param model: model we want to train. After compiling
    :param meta_data: MetaFit class object containing the arguments to use in the model.fit() function
    :param epochs: total number of epochs
    :param epochs_per_iter: number of epochs per iteration
    :param train_iterator: training set iterator
    :param validation_iterator: validation set iterator
    :param callbacks: callbacks
    :param cooldown_time: cooldown time in minutes.
    :param dir: Path to directory in which we want to save our model progress.
        default is dir='.' is just the main directory of the project
    :return: A number of history files saved in the required directory
    """
    # Make directory if not existent:
    if (dir not in os.listdir()) and dir != '.':
        os.makedirs(dir)

    meta_data.epochs = epochs_per_iter
    time_initial = time()
    for iter in range(int(epochs / epochs_per_iter)):
        print(f'Group {iter+1}/{int(epochs / epochs_per_iter)}')
        # history = model.fit(train_iterator, steps_per_epoch=len(train_iterator), epochs=epochs_per_iter, verbose=2,
        #                      validation_data=validation_iterator,
        #                      validation_steps=len(validation_iterator), callbacks=mc)
        history = model_fit(model, meta_data)



        #       Save model progress in the directory:
        hist_index = iter
        if hist_starting_index != 0:
            hist_index = hist_index + hist_starting_index

        # Saving history:
        filename = 'history' + str(hist_index)
        save_model_history(history, filename, dir=dir)
        # Saving model parameters:
        path_latest_model = dir + '/' + 'model_latest_parameters.h5'
        model.save_weights(filepath=path_latest_model)

        # If not last iteration, take a break to cooldown GPU
        if iter != len(range(int(epochs / epochs_per_iter))):
            sleep(cooldown_time*60)

    print(f'Total runtime: {(time()-time_initial)/60:.2f} minutes.')
    print(f'Total cooldown time of {cooldown_time * (int(epochs / epochs_per_iter) - 1) :.2f} minutes was included.')



def save_model_history(history, name, dir=None):
    """
    :param history: History object obtained by history = model.fit(...)
    :param name:str Name of file we wish to save as
    :param dir:path Directory where we want to save our history object. in string format
    :return: saves the file in the specified location, in json format.
    Use "load_history_into_dict()" to load the file again.
    """
    if not (dir is None):  # If we were given directory in which we want to save the model history
        name = dir + '/' + name
    hist_df = pd.DataFrame(history.history)
    hist_json_file = name + '.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)


def load_model_history(file_path):
    with open(file_path) as f:
        history = json.load(f)
    return history


def load_history_into_dict(filepath):
    hist = load_model_history(filepath)
    loss = np.array(list(hist.get('loss').values()))
    mean_iou = np.array(list(hist.get('my_mean_iou').values()))
    val_loss = np.array(list(hist.get('val_loss').values()))
    val_mean_iou = np.array(list(hist.get('val_my_mean_iou').values()))
    hist_dict = dict([('loss', loss), ('my_mean_iou', mean_iou), ('val_loss', val_loss), ('val_my_mean_iou', val_mean_iou)])
    return hist_dict


def merge_histories_in_directory(dir, name='merged_history'):
    """
    This function is used to merge all of the history objects saved in a directory to a single history object.
    It will save a file named 'merged_history.json' in the same directory.
    :returns: Merged history dictionary with the following keys: 'loss', 'my_mean_iou', 'val_loss', 'val_my_mean_iou'.
        The values of these keys are numpy nd.arrays .
    """
    dirFiles = natsort.natsorted(os.listdir(dir))
    losses, mean_ious, val_losses, val_mean_ious = [], [], [], []
    for file in dirFiles:
        if file.endswith('.json') and file.startswith('history'):
            hist = load_history_into_dict(dir + '/' + file)
            losses.append(hist.get('loss'))
            mean_ious.append(hist.get('my_mean_iou'))
            val_losses.append(hist.get('val_loss'))
            val_mean_ious.append(hist.get('val_my_mean_iou'))
    # Merging the values independently:
    losses = np.concatenate(losses)
    mean_ious = np.concatenate(mean_ious)
    val_losses = np.concatenate(val_losses)
    val_mean_ious = np.concatenate(val_mean_ious)
    # Setting as a dictionary:
    merged_history = dict([('loss', losses), ('my_mean_iou', mean_ious),
                           ('val_loss', val_losses), ('val_my_mean_iou', val_mean_ious)])
    # Saving it into a file with a default name 'merged_history' in the same directory:
    merged_history_df = pd.DataFrame(merged_history)
    merged_hist_json_file = dir + '/' + name + '.json'
    with open(merged_hist_json_file, mode='w') as f:
        merged_history_df.to_json(f)

    return merged_history


def count_units(model):  # This function counts the number of units in a model.
    tot_out = 0
    out_list = []
    for lyr in model.layers:
        if lyr.trainable:
            # This is to tackle any layers that have the output shape as a list of tuples (e.g Input layer)
            if isinstance(lyr.output_shape, list):
                curr_out = reduce(operator.mul, chain(*[s[1:] for s in lyr.output_shape]), 1)
            # This is to tackle other layers like Dense and Conv2D
            elif isinstance(lyr.output_shape, tuple):
                curr_out = reduce(operator.mul, lyr.output_shape[1:], 1)
            else:
                raise TypeError
            tot_out += curr_out
            out_list.append(curr_out)
    return tot_out, out_list


def required_memory(model, batch_size=1, input_shape=(200, 200, 3)):
    """
    This method is for evaluating the VRAM required for running a model.
    :param model: a model to evaluate for. Should be after compiling
    :param batch_size: Number of samples within every batch. default is 1. influences VRAM requirements significantly.
    :param input_shape: shape of the image
    :return: Prints out the estimated memory required for running the model, devided into categroies:
        Activations memory - memory required to retain the activations throughout all of the model layers. Influenced
        significantly with the batch size.
        Parmeters memory - memory required to retain the trainable parameters of the model.
        Miscellainious memory - memory required to retain the batch data itself.
        If we use optimizers with "memory" such as Adam, Rmsprop or Adagrad, the total memory required is
        up to 3 times larger.
    """
    opt = model.optimizer._name
    num_activations, _ = count_units(model)
    num_params = model.count_params()
    misc = input_shape[0] * input_shape[1] * input_shape[2]
    activations_mem = (num_activations * 2) * 4 * batch_size
    params_mem = num_params * 4
    misc_mem = misc * batch_size * 4
    if opt in ['Adam', 'RMSprop', 'Adagrad']:
        (params_mem, activations_mem, misc_mem) = (3 * params_mem, 3 * activations_mem, 3 * misc_mem)
    total_memory = (activations_mem + params_mem + misc_mem) / 1024 / 1024
    print(f'The total required memory for this model to operate is approximately {total_memory:.2f} MB'.format())
    print(f'Activations memory: {activations_mem / 1024 / 1024:.2f} MB'.format())
    print(f'Parameters memory: {params_mem / 1024 / 1024:.2f} MB'.format())
    print(f'Miscellanious memory: {misc_mem / 1024 / 1024:.2f} MB'.format())


def clear_model_and_garbage(model):
    del model
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()

