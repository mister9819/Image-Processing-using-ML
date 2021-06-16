from tensorflow import keras
import utils
import custom_layers
import copy


def encoder_choice(image_shape, encode_len, encoder_type, layers, inp):
    e = None
    if encoder_type == 'D':
        if inp:
            layers = otherLayers()
        e = EncoderDNN(image_shape, encode_len, layers)
    elif encoder_type == 'C':
        if inp:
            layers = CNNLayers(image_shape, True)
        e = EncoderCNN(image_shape, encode_len, layers)
        layers = helper_layers(image_shape, layers)
    elif encoder_type == 'R' or encoder_type == 'G' or encoder_type == 'L' or \
            encoder_type == 'BR' or encoder_type == 'BG' or encoder_type == 'BL':
        if inp:
            layers = otherLayers()
        if len(layers) == 0:
            print("Cannot have 0 layers")
            exit()
        e = EncoderSpecial(encoder_type, image_shape, encode_len, layers)
    else:
        if not inp:
            print("Wrong choice for Encoder. \nC for CNN \nD for Dense "
                  "\nR for RNN \nG for GRU \nL for LSTM "
                  "\nBR for BiRNN \nG for BiGRU \nL for BiLSTM")
            exit()
    layers.reverse()
    return e, layers


def decoder_choice(image_shape, encode_len, decoder_type, layers, inp):
    d = None
    if decoder_type == 'D':
        d = DecoderDNN(image_shape, encode_len, layers)
    elif decoder_type == 'C':
        return DecoderCNN(image_shape, encode_len, layers)
    elif decoder_type == 'R' or decoder_type == 'G' or decoder_type == 'L' or \
            decoder_type == 'BR' or decoder_type == 'BG' or decoder_type == 'BL':
        if len(layers) == 0:
            print("Cannot have 0 layers")
            exit()
        d = DecoderSpecial(decoder_type, image_shape, encode_len, layers)
    else:
        if not inp:
            print("Wrong choice for Decoder. \nC for CNN \nD for Dense "
                  "\nR for RNN \nG for GRU \nL for LSTM "
                  "\nBR for BiRNN \nG for BiGRU \nL for BiLSTM")
            exit()
    return d


def get_coder_type(keyword):
    types = {"C": "CNN", "D": "Dense", "R": "RNN", "G": "GRU", "L": "LSTM",
             "BR": "Bidirectional RNN", "BG": "Bidirectional GRU", "BL": "Bidirectional LSTM"}
    while True:
        print("Enter choice for " + keyword + ".")
        for key, val in types.items():
            print(key, "for", val)
        coder_type = input("Choice: ")
        if types == "0" or coder_type == "":
            exit()
        coder_type = coder_type.upper()
        if coder_type in types.keys():
            break
    return coder_type


def get_model_name(model_type, encoder_type, decoder_type, encode_len, layers_e, layers_d):
    models = {"custom_mirror": "CM", "custom": "C", "basic_mirror": "B", "basic": "B"}
    name = models[model_type] + "_e_"
    if decoder_type == "C":
        layers_d.reverse()
    layers_e.reverse()
    for l in layers_e:
        if encoder_type == "C":
            name += encoder_type + str(l[0]) + "-" + str(l[1]) + "_"
        else:
            name += encoder_type + str(l) + "_"
    name += "d_"
    for l in layers_d:
        if decoder_type == "C":
            name += decoder_type + str(l[0]) + "-" + str(l[1]) + "_"
        else:
            name += decoder_type + str(l) + "_"
    name += str(encode_len)
    return name


def create_model(model_type, image_shape, encoder_type, decoder_type, encode_len,
                 layers_e=None, layers_d=None, inp=False):
    # Take input for type of models
    if layers_d is None:
        layers_d = []
    if layers_e is None:
        layers_e = []
    if inp:
        encode_len = int(input("Enter length of encoded message: "))

    if model_type == "custom_mirror":
        if inp:
            encoder_type = get_coder_type("encoder")
        e, layers_e = encoder_choice(image_shape, encode_len, encoder_type, layers_e, inp)

        layers_d = copy.deepcopy(layers_e)
        decoder_type = encoder_type
        d = decoder_choice(image_shape, encode_len, decoder_type, layers_d, inp)
        if encoder_type == "C":
            layers_d.reverse()
    elif model_type == "custom":
        if inp:
            encoder_type = get_coder_type("autoencoder")
        if encoder_type == "C":
            layers_e = helper_layers(image_shape, layers_e)
        # Only need the encoder, discard layers
        e, layers_e = encoder_choice(image_shape, encode_len, encoder_type, layers_e, inp)

        if inp:
            decoder_type = get_coder_type("decoder")
            if decoder_type == "C":
                print("\nNote: You'll be making decoder upside down to finally match the image size.")
                layers_d = CNNLayers(image_shape, False)
            elif decoder_type == "D":
                layers_d = otherLayers()
            else:
                layers_d = otherLayers()
                if len(layers_d) == 0:
                    print("Cannot have 0 layers")
                    exit()
        if decoder_type == "C":
            layers_d = helper_layers(image_shape, layers_d)
            layers_d.reverse()
        d = decoder_choice(image_shape, encode_len, decoder_type, layers_d, inp)
        if inp and decoder_type == "C":
            layers_d.reverse()
    elif model_type == "basic_mirror":
        if inp:
            encoder_type = get_coder_type("autoencoder")
        if encoder_type == "C":
            layers_e = helper_layers(image_shape, [28, 56])
        elif encoder_type == "D":
            layers_e = []
        else:
            layers_e = [128, 64]
        e, layers_e = encoder_choice(image_shape, encode_len, encoder_type, layers_e, False)

        layers_d = copy.deepcopy(layers_e)
        decoder_type = encoder_type
        d = decoder_choice(image_shape, encode_len, decoder_type, layers_d, False)
        if encoder_type == "C":
            layers_d.reverse()
    elif model_type == "basic":
        if inp:
            encoder_type = get_coder_type("encoder")
        if encoder_type == "C":
            layers_e = helper_layers(image_shape, [28, 28])
        elif encoder_type == "D":
            layers_e = []
        else:
            layers_e = [128, 64]
        # Only need the encoder, discard layers
        e, layers_e = encoder_choice(image_shape, encode_len, encoder_type, layers_e, False)

        if inp:
            decoder_type = get_coder_type("decoder")
        if decoder_type == "C":
            layers_d = helper_layers(image_shape, [28, 28])
            layers_d.reverse()
        elif decoder_type == "D":
            layers_d = []
        else:
            layers_d = [64, 128]
        d = decoder_choice(image_shape, encode_len, decoder_type, layers_d, False)
    else:
        e = None
        d = None
        print("No such model exists. Pick one from 'basic', 'basic_mirror', 'custom_mirror'")
        exit()

    # define input to the model:
    x = keras.Input(shape=image_shape)
    # make the model:
    name = get_model_name(model_type, encoder_type, decoder_type, encode_len, layers_e, layers_d)
    print(name)
    autoencoder = keras.Model(x, d(e(x)), name=name)

    # compile the model:
    autoencoder.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
    # autoencoder.layers[-2].summary()
    # autoencoder.layers[-1].summary()
    return autoencoder


def EncoderDNN(image_shape, encode_len, layers):
    encoder_input = keras.Input(shape=image_shape, name="EncoderInput")
    x = keras.layers.Flatten()(encoder_input)
    for layer in layers:
        x = keras.layers.Dense(layer, activation="relu")(x)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    return keras.Model(encoder_input, encoder_output, name='EncoderDNN')


def DecoderDNN(image_shape, encode_len, layers):
    decoder_input = keras.Input(shape=(encode_len,), name="DecoderInput")
    if len(layers) == 0:
        decoder_output = custom_layers.DecoderOutput(image_shape)(decoder_input)
    else:
        x = keras.layers.Dense(layers[0], activation="relu")(decoder_input)
        for layer in layers[1:]:
            x = keras.layers.Dense(layer, activation="relu")(x)
        decoder_output = custom_layers.DecoderOutput(image_shape)(x)
    return keras.Model(decoder_input, decoder_output, name='DecoderDNN')


def EncoderCNN(image_shape, encode_len, layers):
    layers = helper_layers(image_shape, layers)

    encoder_input = keras.Input(shape=image_shape, name="EncoderInput")
    x = custom_layers.CNNBlock(layers[0][0], True, layers[0][1])(encoder_input)
    for layer in layers[1:]:
        x = custom_layers.CNNBlock(layer[0], True, layer[1])(x)
    x = keras.layers.Flatten()(x)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    return keras.Model(encoder_input, encoder_output, name='EncoderCNN')


def DecoderCNN(image_shape, encode_len, layers):
    helper_shape = layers[0][2]
    decoder_input = keras.Input(shape=(encode_len,), name="DecoderInput")
    x = keras.layers.Dense(utils.number_of_pixels(helper_shape), activation="relu")(decoder_input)
    x = keras.layers.Reshape(helper_shape)(x)
    for layer in layers:
        x = custom_layers.CNNBlock(layer[0], False, layer[1])(x)
    decoder_output = keras.layers.Conv2D(image_shape[2], 3, activation='relu', padding='same')(x)
    return keras.Model(decoder_input, decoder_output, name='DecoderCNN')


def EncoderSpecial(layer_type, image_shape, encode_len, layers):
    encoder_input = keras.Input(shape=(None, image_shape[0]), name="EncoderInput")
    if len(layers) == 1:
        x = specialLayer(layer_type, layers[0], False)(encoder_input)
    else:
        x = specialLayer(layer_type, layers[0], True)(encoder_input)
        for layer in layers[1:-1]:
            x = specialLayer(layer_type, layer, True)(x)
        x = specialLayer(layer_type, layers[-1], False)(x)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    return keras.Model(encoder_input, encoder_output, name='Encoder' + layer_type)


def DecoderSpecial(layer_type, image_shape, encode_len, layers):
    decoder_input = keras.Input(shape=(encode_len,), name="DecoderInput")
    x = keras.layers.Reshape((encode_len, 1))(decoder_input)
    for layer in layers[:-1]:
        x = specialLayer(layer_type, layer, True)(x)
    x = specialLayer(layer_type, layers[-1], False)(x)
    decoder_output = custom_layers.DecoderOutput(image_shape)(x)
    return keras.Model(decoder_input, decoder_output, name='Decoder' + layer_type)


def specialLayer(layer_type, neurons, return_sequences, activation='tanh'):
    if layer_type == 'R':
        return keras.layers.SimpleRNN(neurons, return_sequences=return_sequences, activation=activation)
    elif layer_type == 'G':
        return keras.layers.GRU(neurons, return_sequences=return_sequences, activation=activation)
    elif layer_type == 'L':
        return keras.layers.LSTM(neurons, return_sequences=return_sequences, activation=activation)
    elif layer_type == 'BR':
        return keras.layers.Bidirectional(
            keras.layers.SimpleRNN(neurons, return_sequences=return_sequences, activation=activation)
        )
    elif layer_type == 'BG':
        return keras.layers.Bidirectional(
            keras.layers.GRU(neurons, return_sequences=return_sequences, activation=activation)
        )
    elif layer_type == 'BL':
        return keras.layers.Bidirectional(
            keras.layers.LSTM(neurons, return_sequences=return_sequences, activation=activation)
        )
    else:
        print("Wrong type entered")


def otherLayers():
    layers = []
    while True:
        choice = input("\n0 -> Done \nn -> Number of neurons \nn: ")
        if choice == '0' or choice == '':
            break
        elif choice.isdigit():
            choice = int(choice)
            layers.append(choice)
            print("\nCurrent layers:")
            for i, layer in enumerate(layers):
                print(str(i + 1) + ". Neurons:", layer)
        else:
            print("Invalid response.")

    return layers


def virtual_shape(shape, features, pool, down_sample):
    if down_sample:
        if shape[0] % pool == 0:
            return True, (int(shape[0]/pool), int(shape[1]/pool), features)
        else:
            return False, shape


def helper_layers(image_shape, layers):
    nlayers = []
    if len(layers) == 0:
        print("Wrong format. Use [f1, f2, f3] for default pool as 2 or [[f1, p1], [f2, p2], [f3, p3]] for layers")
        exit()
    temp_shape = image_shape
    for layer in layers:
        if isinstance(layer, int):
            correct, temp_shape = virtual_shape(temp_shape, layer, 2, True)
            if not correct:
                print("Wrong layers made.")
                exit()
            nlayers.append([layer, 2, temp_shape])
        elif len(layer) == 2:
            correct, temp_shape = virtual_shape(temp_shape, layer[0], layer[1], True)
            if not correct:
                print("Wrong layers made.")
                exit()
            nlayers.append([layer[0], layer[1], temp_shape])
        elif len(layer) == 3:
            return layers
        else:
            print("Wrong format. Use [f1, f2, f3] for default pool as 2 or [[f1, p1], [f2, p2], [f3, p3]] for layers")
            exit()
    return nlayers


def CNNLayers(image_shape, isEncoder):
    temp_shape = image_shape
    layers = []
    if isEncoder:
        keyword = "pool"
    else:
        keyword = "downsample"
    while True:
        choice = input("Current ouput: " + str(temp_shape) + "\n0. Done \n1. Add layer \nChoice: ")
        if choice == '1' or choice == '':
            features = input("Enter number of features: ")
            if features == "":
                features = image_shape[0]
            elif features.isdigit():
                features = int(features)
            else:
                print("Invalid choice.")
                continue
            pool = input("Enter size of " + keyword + ": ")
            if pool == "":
                pool = 2
            elif pool.isdigit():
                pool = int(pool)
            else:
                print("Invalid choice.")
                continue
            correct, temp_shape = virtual_shape(temp_shape, features, pool, True)
            if not correct:
                c = input("Cannot " + keyword + " into. 1 - Continue, 0 - End with current layers")
                if c == '1':
                    continue
                else:
                    break
            layers.append([features, pool, temp_shape])
            print("\nCurrent layers:")
            if isEncoder:
                for i, layer in enumerate(layers):
                    print(str(i+1) + ". Features:", layer[0], keyword + ":", layer[1])
            else:
                for i, layer in enumerate(reversed(layers)):
                    print(str(i+1) + ". Features:", layer[0], keyword + ":", layer[1])
        else:
            break

    if len(layers) == 0:
        print("Cannot have 0 layers.")
        exit()
    return layers
