import matplotlib.pyplot as plt
import pickle
import os

import numpy
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import tensorflow.keras.backend as K
import csv

#================== Tensorflow and models ==================#
def use_GPU():
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def test_accuracy(model, x_test):
    test_loss, test_acc = model.evaluate(x_test, x_test, verbose=0)
    return test_loss, test_acc


def model_predict(model, test):
    shape = np.shape(test)
    if len(shape) == 2:
        shape = append_element_tuple(shape)
    ae_out = model.predict(test.reshape(-1, shape[0], shape[1], shape[2]))
    return ae_out[0]


class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.epochs = []
        self.timetaken = tf.timestamp()
    def on_epoch_end(self,epoch, logs = {}):
        self.times.append(tf.timestamp() - self.timetaken)
        self.epochs.append(epoch)
    def on_train_end(self, logs = {}):
        global times
        times = []
        for i in range(len(self.epochs)):
            times.append(self.times[i].numpy())
        times.append(len(self.epochs))


def train_model_history(model, x_train, epochs=3, batch_size=32, inp=False):
    history = model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, callbacks=[timecallback()])
    history.history['time'] = times[:-1]
    history.history['epochs'] = times[-1]
    return model, history.history


def train_model(model, x_train, epochs=3, batch_size=32, validation_split=0.1, inp=False):
    model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return model


def test_train_model(model, x_train, epochs=3, batch_size=32, validation_split=0.1, inp=False):
    try:
        model1 = model = tf.keras.models.load_model("models/" + model.name + ".model", custom_objects={'SSIMLoss': SSIMLoss})
        if inp:
            while True:
                load = pinput("Trained model already exists, load? (y/n): ", 'blue')
                if load == 'y':
                    model = model1
                    while True:
                        load = pinput("Train the model more? (y/n): ", 'blue')
                        if load == 'y':
                            break
                        elif load == 'n':
                            return model
                    break
                elif load == 'n':
                    break
    except IOError:
        pass
    model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    save_model(model)
    return model

def save_model(model, name=None):
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, 'models')
    if not (os.path.exists(final_directory)) or len(os.listdir(final_directory)) == 0:
        os.makedirs(final_directory)
    if name == None:
        name = model.name
    try:
        model.save("models/" + name + ".model")
    except:
        pass
    pprint("Model saved.", "green")


def load_model(name):
    model = tf.keras.models.load_model("models/" + name + ".model", custom_objects={'SSIMLoss': SSIMLoss})
    pprint("Model loaded.", 'green')
    return model


def get_encoder(model):
    return model.layers[-2]


def get_decoder(model):
    return model.layers[-1]


def load_model_options():
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "models")
    try:
        models = os.listdir(final_directory)
        pprint("Pick an index to load model", 'blue')
        pprint("0. Exit", 'blue')
        for i in range(0, len(models)):
            pprint("{str(i+1)}. {models[i]}", 'blue')
        while True:
            choice = int(input("Choice: "))
            if choice == 0:
                exit()
            if choice > 0 and choice <= len(models):
                return load_model(models[choice-1][:-3])
            pprint("Wrong Choice", 'fail')
    except:
        pprint("No models exist yet.", 'fail')
        exit()


def get_trainable_params(model):
    return np.sum([K.count_params(w) for w in model.trainable_weights])


def get_non_trainable_params(model):
    return np.sum([K.count_params(w) for w in model.non_trainable_weights])


def get_total_params(model):
    return (np.sum([K.count_params(w) for w in model.non_trainable_weights])
            + np.sum([K.count_params(w) for w in model.trainable_weights]))


def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

#================== Measuring Performance ==================#

#================== Image Augmentation ==================#
def flip(data):
    d1 = tf.image.flip_up_down(data)
    d2 = tf.image.flip_left_right(data)
    data = np.concatenate((data, d1, d2))
    return data

def rotate(data):
    d1 = tf.image.rot90(data, k=1)
    d2 = tf.image.rot90(data, k=2)
    data = np.concatenate((data, d1, d2))
    return data

def keras_augmentation(x_train):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    x_train1 = np.copy(x_train)
    train_datagen.fit(x_train)
    x_train = np.concatenate((x_train, x_train1))
    x_train1 = 0

    return x_train


#================== Datasets ==================#
def save_data(name, data):
    pickle.dump(data, open(name + ".p", "wb"))


def load_data(name):
    data = pickle.load(open(name + ".p", "rb"))
    return data


def generate_dataset(name, train=True):
    if train:
        split = 'train'
    else:
        split = 'test'
    try:
        image, label = tfds.as_numpy(tfds.load(
            name,
            split=split,
            batch_size=-1,
            as_supervised=True,
            data_dir="data/cache/"
        ))
    except:
        image = 0
    return image


def get_data(names, test=False):
    train_data = []
    test_data = []
    shape = None

    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, 'data')
    if not (os.path.exists(final_directory)) or len(os.listdir(final_directory)) == 0:
        os.makedirs(final_directory)

    for name in names:
        if name in tfds.list_builders():
            directory = r'data/' + name
            current_directory = os.getcwd()
            final_directory = os.path.join(current_directory, directory)
            if not(os.path.exists(final_directory)) or len(os.listdir(final_directory)) == 0:
                os.makedirs(final_directory)
                x_train = generate_dataset(name)
                if not(isinstance(x_train, int)):
                    save_data(directory + '/' + name, x_train)
                x_test = generate_dataset(name, train=False)
                if not(isinstance(x_test, int)):
                    save_data(directory + '/' + name + 't', x_test)
                # shutil.rmtree(current_directory + '\\data\\cache')
            temp_train_data = load_data(directory + '/' + name)
            if shape == None or shape == temp_train_data[-1].shape:
                pprint(f"{name} added to data.", 'GREEN')
                train_data.extend(temp_train_data)
                temp_train_data = None
                if not test:
                    return train_data
                test_data.extend(load_data(directory + '/' + name + 't'))
                shape = train_data[-1].shape
            else:
                pprint("Datasets cannot be combined as their shapes don't match.", "FAIL")
                exit()
        else:
            pprint(f"Dataset {name} does not exist.", "FAIL")
    return np.array(train_data), np.array(test_data)


def generate_csv(data):
    fields = ['Name', 'Encoder Train Params', 'Encoder Non Train Params', 'Encoder All Params',
              'Decoder Train Params', 'Decoder Non Train Params', 'Decoder All Params',
              'Encoded Message Length', 'Compression Ratio', 'Loss', 'Accuracy', 'Time to Train',
              'Epochs']
    try:
        with open('history.csv', 'x') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(fields)
            csvwriter.writerows(data)
            print("Data written successfully to history.csv")
    except FileExistsError:
        with open('history.csv', 'a') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerows(data)
            print("Data written successfully to history.csv")

#================== General Utils ==================#
def append_element_tuple(t, e=1):
    t = list(t)
    t.append(1)
    t = tuple(t)
    return t


def random_test(test):
    index = np.random.choice(test.shape[0], replace=False)
    return test[index]


#================== Plotting Graphs ==================#
def plot2(fig1, fig2):
    fig = plt.figure()

    sub_plot = fig.add_subplot(1, 2, 1)
    sub_plot.set_title("Original")
    plt.imshow(fig1, cmap="gray")

    sub_plot = fig.add_subplot(1, 2, 2)
    sub_plot.set_title("Encoded -> Decoded")
    plt.imshow(fig2, cmap="gray")

    plt.show()


#================== Image handling ==================#
def image_shape_from_data(x):
    shape = np.shape(x)[1:]
    if len(shape) == 2:
        shape = append_element_tuple(shape)
    return shape


def image_shape(x):
    shape = np.shape(x)
    return shape


def number_of_pixels(shape):
    ans = 1
    for i in shape:
        ans = ans * i
    return ans


def split_image_data(data):
    ndata = []
    for image in data:
        for i in range(3):
            ndata.append(image[:, :, i])
    ndata = numpy.array(ndata)
    return ndata[..., np.newaxis]


def merge_image(r, g, b):
    return np.dstack((r, g, b))


#================== Color Printing ==================#
styles = {'HEADER': '\033[95m', 'BLUE': '\033[94m', 'CYAN': '\033[96m', 'GREEN': '\033[92m',
              'WARNING': '\033[93m', 'FAIL': '\033[91m', 'ENDC': '\033[0m', 'BOLD': '\033[1m',
              'UNDERLINE': '\033[4m'}


def pprint(text, style):
    style = style.upper()
    if style in styles:
        print(f"{styles[style]}{text}{styles['ENDC']}")
    else:
        print(f"{styles['FAIL']}Wrong style chosen.{styles['ENDC']}")

def pinput(text, style):
    style = style.upper()
    if style in styles:
        var = input(f"{styles[style]}{text}{styles['ENDC']}")
        return var
    else:
        print(f"{styles['FAIL']}Wrong style chosen.{styles['ENDC']}")