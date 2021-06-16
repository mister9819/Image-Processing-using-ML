from tensorflow import keras
import utils
import custom_layers


def encoder_choice(image_shape, encode_len, encoder_type, layers, inp):
    e = None
    if encoder_type == 'D':
        if inp:
            layers = otherLayers()
        e = EncoderDNN(image_shape, encode_len, layers)
    elif encoder_type == 'C':
        if inp:
            layers = CNNLayers(image_shape, True)
        else:
            layers = helper_layers(image_shape, layers)
        e = EncoderCNN(image_shape, encode_len, layers)
    elif encoder_type == 'R' or encoder_type == 'G' or encoder_type == 'L' or \
            encoder_type == 'BR' or encoder_type == 'BG' or encoder_type == 'BL':
        if inp:
            layers = otherLayers()
        if len(layers) == 0:
            print("Cannot have 0 layers")
            exit()
        e = EncoderSpecial(encoder_type, image_shape, encode_len, layers)
    else:
        if not(inp):
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
        if not (inp):
            print("Wrong choice for Decoder. \nC for CNN \nD for Dense "
                  "\nR for RNN \nG for GRU \nL for LSTM "
                  "\nBR for BiRNN \nG for BiGRU \nL for BiLSTM")
            exit()
    return d


def create_model(image_shape, encoder_type, decoder_type, encode_len, layers=[], inp=False):
    # Take input for type of models
    if inp:
        encode_len = int(input("Enter length of encoded message: "))
        while True:
            encoder_type = input("Enter choice for autoencoder. \nC for CNN \nD for Dense "
              "\nR for RNN \nG for GRU \nL for LSTM "
              "\nBR for BiRNN \nBG for BiGRU \nBL for BiLSTM \n0 for exit \nChoice: ")
            if encoder_type == "0":
                exit()
            encoder_type = encoder_type.upper()

            e, layers = encoder_choice(image_shape, encode_len, encoder_type, [], inp)

            decoder_type = encoder_type
            d = decoder_choice(image_shape, encode_len, decoder_type, layers, inp)
            if d != None:
                break
    else:
        e, layers = encoder_choice(image_shape, encode_len, encoder_type, layers, inp)
        d = decoder_choice(image_shape, encode_len, decoder_type, layers, inp)

    # define input to the model:
    x = keras.Input(shape=image_shape)
    # make the model:
    name = "e_" + encoder_type + "_d_" + decoder_type + "_" + str(encode_len)
    autoencoder = keras.Model(x, d(e(x)), name=name)

    # compile the model:
    autoencoder.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
    autoencoder.layers[-2].summary()
    autoencoder.layers[-1].summary()
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


def EncoderSpecial(type, image_shape, encode_len, layers):
    encoder_input = keras.Input(shape=(None, image_shape[0]), name="EncoderInput")
    if len(layers) == 1:
        x = specialLayer(type, layers[0], False)(encoder_input)
    else:
        x = specialLayer(type, layers[0], True)(encoder_input)
        for layer in layers[1:-1]:
            x = specialLayer(type, layer, True)(x)
        x = specialLayer(type, layers[-1], False)(x)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    return keras.Model(encoder_input, encoder_output, name='Encoder' + type)


def DecoderSpecial(type, image_shape, encode_len, layers):
    decoder_input = keras.Input(shape=(encode_len,), name="DecoderInput")
    x = keras.layers.Reshape((encode_len, 1))(decoder_input)
    for layer in layers[:-1]:
        x = specialLayer(type, layer, True)(x)
    x = specialLayer(type, layers[-1], False)(x)
    decoder_output = custom_layers.DecoderOutput(image_shape)(x)
    return keras.Model(decoder_input, decoder_output, name='Decoder' + type)


def specialLayer(type, neurons, return_sequences, activation='tanh'):
    if type == 'R':
        return keras.layers.SimpleRNN(neurons, return_sequences=return_sequences, activation=activation)
    elif type == 'G':
        return keras.layers.GRU(neurons, return_sequences=return_sequences, activation=activation)
    elif type == 'L':
        return keras.layers.LSTM(neurons, return_sequences=return_sequences, activation=activation)
    elif type == 'BR':
        return keras.layers.Bidirectional(
            keras.layers.SimpleRNN(neurons, return_sequences=return_sequences, activation=activation)
        )
    elif type == 'BG':
        return keras.layers.Bidirectional(
            keras.layers.GRU(neurons, return_sequences=return_sequences, activation=activation)
        )
    elif type == 'BL':
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
        print("Wrong format. Use [f1, f2, f3] for default pool as 2 or [[f1, p1], [f2, p2], [f3, p3]]")
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
        else:
            print("Wrong format. Use [f1, f2, f3] for default pool as 2 or [[f1, p1], [f2, p2], [f3, p3]]")
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
            for i, layer in enumerate(layers):
                print(str(i+1) + ". Features:", layer[0], keyword + ":", layer[1])
        else:
            break

    if len(layers) == 0:
        print("Cannot have 0 layers.")
        exit()
    return layers