import matplotlib.pyplot as plt
import pickle
import os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import shutil

#================== Tensorflow and models ==================#
def use_GPU():
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def test_accuracy(model, x_test):
    test_loss, test_acc = model.evaluate(x_test, x_test, verbose=2)
    print('\nTest accuracy:', test_acc)


def model_predict(model, test):
    shape = np.shape(test)
    if len(shape) == 2:
        shape = append_element_tuple(shape)
    ae_out = model.predict(test.reshape(-1, shape[0], shape[1], shape[2]))
    return ae_out[0]


def train_model(model, x_train, epochs=3, batch_size=32, validation_split=0.1, inp=False):
    model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return model


def test_train_model(model, x_train, epochs=3, batch_size=32, validation_split=0.1, inp=False):
    try:
        model1 = tf.keras.models.load_model("models/" + model.name + ".h5")
        if inp:
            while True:
                load = input("Trained model already exists, load? (y/n): ")
                if load == 'y':
                    model = model1
                    while True:
                        load = input("Train the model more? (y/n): ")
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
    if name == None:
        name = model.name
    try:
        tf.keras.models.save_model(model, "models/" + name + ".h5")
    except:
        pass
    print("Model saved.")


def load_model(name):
    model = tf.keras.models.load_model("models/" + name + ".h5")
    print("Model loaded.")
    return model


def get_encoder(model):
    return model.layers[-2]


def get_decoder(model):
    return model.layers[-1]


def load_model_options():
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "models")
    models = os.listdir(final_directory)
    print("Pick an index to load model")
    print("0. Exit")
    for i in range(0, len(models)):
        print(str(i+1) + ".", models[i])
    while True:
        choice = int(input("Choice: "))
        if choice == 0:
            exit()
        if choice > 0 and choice <= len(models):
            return load_model(models[choice-1][:-3])
        print("Wrong Choice")

# Might have to edit if CNN is made dynamic
def get_helper_shape(image_shape):
    return (int(image_shape[0]/4), int(image_shape[0]/4), image_shape[0])


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


def get_data(name, test=False):
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
        train_data = load_data(directory + '/' + name)
        if not test:
            return train_data
        test_data = load_data(directory + '/' + name + 't')
        return train_data, test_data
    else:
        print("No dataset exists")
        return None


#================== General Utils ==================#
def append_element_tuple(t, e=1):
    t = list(t)
    t.append(1)
    t = tuple(t)
    return t


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