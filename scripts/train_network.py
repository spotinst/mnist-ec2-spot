from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as Keras
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, Flatten
from keras.models import load_model

import time
import requests
import os
import datetime
import glob


def main():
    batch_size = 128
    num_classes = 10
    epochs = 12
    global model

    # input image dimensions
    img_rows, img_cols = 28, 28

    volume_mount_dir = '/dl/'
    dataset_path = os.path.join(volume_mount_dir, 'datasets')
    checkpoint_path = os.path.join(volume_mount_dir, 'checkpoints')
    checkpoint_names = 'mnist_model.{epoch:03d}.h5'
    today_date = datetime.datetime.today().strftime('%Y-%m-%d')

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data(dataset_path + "/mnist.npz")

    if Keras.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Load model
    if not len(os.path.isdir(checkpoint_path)) == 0:
        # checkpoints exist
        model, epoch_number = load_model_from_checkpoints(checkpoint_path, checkpoint_names)
    else:
        # checkpoints not exist - prepare the model from scratch
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        epoch_number = 0

    # create_callbacks_functions Callbacks:
    # A callback is a set of functions to be applied at given stages of the training procedure.
    # You can use callbacks to get a view on internal states and statistics of the model during training.
    checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_names)
    callbacks = create_callbacks_functions(volume_mount_dir, checkpoint_filepath, today_date)

    # prepare model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # train model
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              initial_epoch=epoch_number,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=callbacks)

    # evaluate model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    save_model(today_date)


def load_model_from_checkpoints(path, name):
    # loads all the checkpoints paths from 'checkpoint_path'
    checkpoint_files = glob.glob(os.path.join(path, '*'))
    # get the latest checkpoint by the epoch no'
    checkpoint_epoch_number = max([int(file.split(".")[1]) for file in checkpoint_files])
    checkpoint_epoch_path = os.path.join(path,
                                         name.format(epoch=checkpoint_epoch_number))

    # load model from checkpoint
    checkpoint_model = load_model(checkpoint_epoch_path)

    return checkpoint_model, checkpoint_epoch_number


def get_spot_termination_callback():
    class SpotTermination(keras.callbacks.Callback):
        def on_batch_begin(self, batch, logs={}):
            status_code = requests.get("http://169.254.169.254/latest/meta-data/spot/instance-action").status_code
            if status_code != 404:
                time.sleep(150)

    return SpotTermination()


def create_callbacks_functions(volume_mount_dir, checkpoint_file_path, today_date):
    # Create checkpoints callback
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file_path,
                                          save_weights_only=False,
                                          monitor='val_loss')
    # Loss history callback
    epoch_results_callback = CSVLogger(os.path.join(volume_mount_dir, 'log_{}.csv'.format(today_date)),
                                       append=True)
    # Create spot termination callback
    spot_termination_callback = get_spot_termination_callback()
    callbacks = [checkpoint_callback, epoch_results_callback, spot_termination_callback]

    return callbacks


def save_model(today_date):
    # saving the model
    # serialize model to JSON
    model_json = model.to_json()
    with open("/dl/models/model_mnist_{}.json".format(today_date), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("/dl/models/model_mnist_{}.h5".format(today_date))
    print("Saved model to disk")


if __name__ == "__main__":
    main()


