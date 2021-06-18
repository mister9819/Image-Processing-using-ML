import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import models
import utils

def run():
    utils.use_GPU()

    x_train, x_test = utils.get_data("mnist", test=True)
    x_train = x_train / 255.0
    image_shape = utils.image_shape_from_data(x_train)
    data = []

    # Edit to train multiple models
    choice = "3"
    debug = True
    autoencoder_types = [["C", "C"]]
    autoencoder_layers = [[[28, 28], [28, 28]], [[28, 56], [56, 28]]]
    encode_lens = [64]

    if not debug:
        print("0. Exit")
        print("1. Create new basic model")
        print("2. Create new basic mirror model")
        print("3. Create new custom model")
        print("4. Create new custom mirror model")
        print("5. Load existing model")
        choice = input("Choice: ")
    model = None

    for autoencoder_type in autoencoder_types:
        for autoencoder_layer in autoencoder_layers:
            for encode_len in encode_lens:
                if choice == "0" or choice == "":
                    exit()
                elif choice == "5":
                    model = utils.load_model_options()
                    print("0. Exit \n1. Don't train loaded model \n2. Train loaded model")
                    choice = int(input("Choice: "))
                    if choice == 0:
                        exit()
                    elif choice == 1:
                        pass
                    elif choice == 2:
                        model = utils.train_model(model, x_train)
                else:
                    keyword = ""
                    if choice == "1":
                        keyword = "basic"
                    elif choice == "2":
                        keyword = "basic_mirror"
                    elif choice == "3":
                        keyword = "custom"
                    elif choice == "4":
                        keyword = "custom_mirror"
                    else:
                        print("Wrong choice.")
                        exit()
                    model = models.create_model(keyword, image_shape, autoencoder_type[0], autoencoder_type[1],
                                                encode_len, autoencoder_layer[0], autoencoder_layer[1], inp=False)
                    model, history = utils.train_model_history(model, x_train, epochs=2, inp=False)

                encoder = utils.get_encoder(model)
                decoder = utils.get_decoder(model)
                data.append([model.name, utils.get_trainable_params(encoder), utils.get_non_trainable_params(encoder),
                             utils.get_total_params(encoder), utils.get_trainable_params(decoder),
                             utils.get_non_trainable_params(decoder), utils.get_total_params(decoder), encode_len,
                             utils.number_of_pixels(utils.image_shape(x_train[0]))/encode_len, history['loss'][-1],
                             history['accuracy'][-1], history['time'][-1], history['epochs']])

    utils.generate_csv(data)