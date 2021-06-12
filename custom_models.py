from tensorflow import keras
import utils
import custom_layers


def encoder_choice(image_shape, encode_len, encoder_type, inp):
    e = None
    if encoder_type == 'D':
        e = EncoderDNN(image_shape, encode_len)
    elif encoder_type == 'C':
        return EncoderCNN(image_shape, encode_len)
    elif encoder_type == 'R':
        e = EncoderRNN(image_shape, encode_len)
    elif encoder_type == 'G':
        e = EncoderGRU(image_shape, encode_len)
    elif encoder_type == 'L':
        e = EncoderLSTM(image_shape, encode_len)
    elif encoder_type == 'BR':
        e = EncoderBiRNN(image_shape, encode_len)
    elif encoder_type == 'BG':
        e = EncoderBiGRU(image_shape, encode_len)
    elif encoder_type == 'BL':
        e = EncoderBiLSTM(image_shape, encode_len)
    else:
        if not(inp):
            print("Wrong choice for Encoder. \nC for CNN \nD for Dense "
                  "\nR for RNN \nG for GRU \nL for LSTM "
                  "\nBR for BiRNN \nG for BiGRU \nL for BiLSTM")
            exit()
    return e


def custom_decoder_choice(image_shape, encode_len, decoder_type, layers, inp):
    d = None
    if decoder_type == 'D':
        d = DecoderDNN(image_shape, encode_len, layers)
    elif decoder_type == 'C':
        return DecoderCNN(image_shape, encode_len, layers)
    elif decoder_type == 'R':
        d = DecoderRNN(image_shape, encode_len)
    elif decoder_type == 'G':
        d = DecoderGRU(image_shape, encode_len)
    elif decoder_type == 'L':
        d = DecoderLSTM(image_shape, encode_len)
    elif decoder_type == 'BR':
        d = DecoderBiRNN(image_shape, encode_len)
    elif decoder_type == 'BG':
        d = DecoderBiGRU(image_shape, encode_len)
    elif decoder_type == 'BL':
        d = DecoderBiLSTM(image_shape, encode_len)
    else:
        if not (inp):
            print("Wrong choice for Decoder. \nC for CNN \nD for Dense "
                  "\nR for RNN \nG for GRU \nL for LSTM "
                  "\nBR for BiRNN \nG for BiGRU \nL for BiLSTM")
            exit()
    return d


def create_model(image_shape, encoder_type, decoder_type, encode_len, inp=False):
    # Take input for type of models
    if inp:
        encode_len = int(input("Enter length of encoded message: "))
        while True:
            encoder_type = input("Enter choice for autoencoder. \nC for CNN \nD for Dense "
                                 "\nChoice: ")
            if encoder_type == "0":
                exit()
            encoder_type = encoder_type.upper()
            e, layers = encoder_choice(image_shape, encode_len, encoder_type, inp)

            decoder_type = encoder_type
            d = custom_decoder_choice(image_shape, encode_len, decoder_type, layers, inp)
            if d != None:
                break
    else:
        e, layers = encoder_choice(image_shape, encode_len, encoder_type, inp)
        print(layers)
        d = custom_decoder_choice(image_shape, encode_len, decoder_type, layers, inp)

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


def EncoderDNN(image_shape, encode_len):
    layers = otherLayers()
    encoder_input = keras.Input(shape=image_shape, name="EncoderInput")
    x = keras.layers.Flatten()(encoder_input)
    for layer in layers:
        x = keras.layers.Dense(layer, activation="relu")(x)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    layers.reverse()
    return keras.Model(encoder_input, encoder_output, name='EncoderDNN'), layers


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


def EncoderCNN(image_shape, encode_len):
    # Build Model
    layers = CNNLayers(image_shape, True)

    # Create Model
    encoder_input = keras.Input(shape=image_shape, name="EncoderInput")
    x = custom_layers.CNNBlock(layers[0][0], True, layers[0][1])(encoder_input)
    for layer in layers[1:]:
        x = custom_layers.CNNBlock(layer[0], True, layer[1])(x)
    x = keras.layers.Flatten()(x)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    layers.reverse()
    return keras.Model(encoder_input, encoder_output, name='EncoderCNN'), layers


def DecoderCNN(image_shape, encode_len, layers):
    helper_shape = layers[0][2]
    decoder_input = keras.Input(shape=(encode_len,), name="DecoderInput")
    x = keras.layers.Dense(utils.number_of_pixels(helper_shape), activation="relu")(decoder_input)
    x = keras.layers.Reshape(helper_shape)(x)
    for layer in layers:
        x = custom_layers.CNNBlock(layer[0], False, layer[1])(x)
    decoder_output = keras.layers.Conv2D(image_shape[2], 3, activation='relu', padding='same')(x)
    return keras.Model(decoder_input, decoder_output, name='DecoderCNN')


def EncoderRNN(image_shape, encode_len):
    encoder_input = keras.Input(shape=(None, image_shape[0]), name="EncoderInput")
    x = keras.layers.SimpleRNN(128, return_sequences=True, activation="relu")(encoder_input)
    x = keras.layers.SimpleRNN(64, activation="relu")(x)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    return keras.Model(encoder_input, encoder_output, name='EncoderRNN')


def DecoderRNN(image_shape, encode_len):
    decoder_input = keras.Input(shape=(encode_len,), name="DecoderInput")
    x = keras.layers.Reshape((encode_len, 1))(decoder_input)
    x = keras.layers.SimpleRNN(64, return_sequences=True, activation="relu")(x)
    x = keras.layers.SimpleRNN(128, activation="relu")(x)
    x = keras.layers.Dense(utils.number_of_pixels(image_shape), activation="relu", name="EncoderOutput")(x)
    decoder_output = keras.layers.Reshape(image_shape)(x)
    return keras.Model(decoder_input, decoder_output, name='DecoderRNN')


def EncoderGRU(image_shape, encode_len):
    encoder_input = keras.Input(shape=(None, image_shape[0]), name="EncoderInput")
    x = keras.layers.GRU(128, return_sequences=True, activation="relu")(encoder_input)
    x = keras.layers.GRU(64, activation="relu")(x)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    return keras.Model(encoder_input, encoder_output, name='EncoderGRU')


def DecoderGRU(image_shape, encode_len):
    decoder_input = keras.Input(shape=(encode_len,), name="DecoderInput")
    x = keras.layers.Reshape((encode_len, 1))(decoder_input)
    x = keras.layers.GRU(64, return_sequences=True, activation="relu")(x)
    x = keras.layers.GRU(128, activation="relu")(x)
    x = keras.layers.Dense(utils.number_of_pixels(image_shape), activation="relu", name="EncoderOutput")(x)
    decoder_output = keras.layers.Reshape(image_shape)(x)
    return keras.Model(decoder_input, decoder_output, name='DecoderGRU')


def EncoderLSTM(image_shape, encode_len):
    encoder_input = keras.Input(shape=(None, image_shape[0]), name="EncoderInput")
    x = keras.layers.LSTM(128, return_sequences=True, activation="relu")(encoder_input)
    x = keras.layers.LSTM(64, activation="relu")(x)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    return keras.Model(encoder_input, encoder_output, name='EncoderLSTM')


def DecoderLSTM(image_shape, encode_len):
    decoder_input = keras.Input(shape=(encode_len,), name="DecoderInput")
    x = keras.layers.Reshape((encode_len, 1))(decoder_input)
    x = keras.layers.LSTM(64, return_sequences=True, activation="relu")(x)
    x = keras.layers.LSTM(128, activation="relu")(x)
    x = keras.layers.Dense(utils.number_of_pixels(image_shape), activation="relu", name="EncoderOutput")(x)
    decoder_output = keras.layers.Reshape(image_shape)(x)
    return keras.Model(decoder_input, decoder_output, name='DecoderLSTM')


def EncoderBiRNN(image_shape, encode_len):
    encoder_input = keras.Input(shape=(None, image_shape[0]), name="EncoderInput")
    x = keras.layers.Bidirectional(
        keras.layers.SimpleRNN(128, return_sequences=True, activation="relu")
    )(encoder_input)
    x = keras.layers.Bidirectional(
        keras.layers.SimpleRNN(64, activation="relu")
    )(x)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    return keras.Model(encoder_input, encoder_output, name='EncoderBiRNN')


def DecoderBiRNN(image_shape, encode_len):
    decoder_input = keras.Input(shape=(encode_len,), name="DecoderInput")
    x = keras.layers.Reshape((encode_len, 1))(decoder_input)
    x = keras.layers.Bidirectional(
        keras.layers.SimpleRNN(64, return_sequences=True, activation="relu")
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.SimpleRNN(128, activation="relu")
    )(x)
    x = keras.layers.Dense(utils.number_of_pixels(image_shape), activation="relu", name="EncoderOutput")(x)
    decoder_output = keras.layers.Reshape(image_shape)(x)
    return keras.Model(decoder_input, decoder_output, name='DecoderBiRNN')


def EncoderBiGRU(image_shape, encode_len):
    encoder_input = keras.Input(shape=(None, image_shape[0]), name="EncoderInput")
    x = keras.layers.Bidirectional(
        keras.layers.GRU(128, return_sequences=True, activation="relu")
    )(encoder_input)
    x = keras.layers.Bidirectional(
        keras.layers.GRU(64, activation="relu")
    )(x)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    return keras.Model(encoder_input, encoder_output, name='EncoderBiGRU')


def DecoderBiGRU(image_shape, encode_len):
    decoder_input = keras.Input(shape=(encode_len,), name="DecoderInput")
    x = keras.layers.Reshape((encode_len, 1))(decoder_input)
    x = keras.layers.Bidirectional(
        keras.layers.GRU(64, return_sequences=True, activation="relu")
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.GRU(128, activation="relu")
    )(x)
    x = keras.layers.Dense(utils.number_of_pixels(image_shape), activation="relu", name="EncoderOutput")(x)
    decoder_output = keras.layers.Reshape(image_shape)(x)
    return keras.Model(decoder_input, decoder_output, name='DecoderBiGRU')


def EncoderBiLSTM(image_shape, encode_len):
    encoder_input = keras.Input(shape=(None, image_shape[0]), name="EncoderInput")
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, activation="relu")
    )(encoder_input)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, activation="relu")
    )(x)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    return keras.Model(encoder_input, encoder_output, name='EncoderBiLSTM')


def DecoderBiLSTM(image_shape, encode_len):
    decoder_input = keras.Input(shape=(encode_len,), name="DecoderInput")
    x = keras.layers.Reshape((encode_len, 1))(decoder_input)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, activation="relu")
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, activation="relu")
    )(x)
    x = keras.layers.Dense(utils.number_of_pixels(image_shape), activation="relu", name="EncoderOutput")(x)
    decoder_output = keras.layers.Reshape(image_shape)(x)
    return keras.Model(decoder_input, decoder_output, name='DecoderBiLSTM')


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


def CNNLayers(image_shape, encoder):
    temp_shape = image_shape
    layers = []
    if encoder:
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