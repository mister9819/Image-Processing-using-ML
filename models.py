from tensorflow import keras
import utils


def encoder_choice(image_shape, encode_len, encoder_type, inp):
    e = None
    if encoder_type == 'D':
        e = EncoderDNN(image_shape, encode_len)
    elif encoder_type == 'C':
        e = EncoderCNN(image_shape, encode_len)
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


def decoder_choice(image_shape, encode_len, decoder_type, inp):
    d = None
    if decoder_type == 'D':
        d = DecoderDNN(image_shape, encode_len)
    elif decoder_type == 'C':
        d = DecoderCNN(image_shape, encode_len)
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
            encoder_type = input("Enter choice for Encoder. \nC for CNN \nD for Dense "
              "\nR for RNN \nG for GRU \nL for LSTM "
              "\nBR for BiRNN \nBG for BiGRU \nBL for BiLSTM \n0 for exit \nChoice: ")
            if encoder_type == "0":
                exit()
            encoder_type = encoder_type.upper()
            e = encoder_choice(image_shape, encode_len, encoder_type, inp)
            if e != None:
                break
        while True:
            decoder_type = input("Enter choice for Decoder. \nC for CNN \nD for Dense "
              "\nR for RNN \nG for GRU \nL for LSTM "
              "\nBR for BiRNN \nG for BiGRU \nL for BiLSTM \n0 for exit \nChoice: ")
            if decoder_type == "0":
                exit()
            decoder_type = decoder_type.upper()
            d = decoder_choice(image_shape, encode_len, decoder_type, inp)
            if d != None:
                break
    else:
        e = encoder_choice(image_shape, encode_len, encoder_type, inp)
        d = decoder_choice(image_shape, encode_len, decoder_type, inp)

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
    encoder_input = keras.Input(shape=image_shape, name="EncoderInput")
    x = keras.layers.Flatten()(encoder_input)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    return keras.Model(encoder_input, encoder_output, name='EncoderDNN')


def DecoderDNN(image_shape, encode_len):
    decoder_input = keras.Input(shape=(encode_len,), name="DecoderInput")
    x = keras.layers.Dense(utils.number_of_pixels(image_shape), activation="relu")(decoder_input)
    decoder_output = keras.layers.Reshape(image_shape)(x)
    return keras.Model(decoder_input, decoder_output, name='DecoderDNN')


def EncoderCNN(image_shape, encode_len):
    encoder_input = keras.Input(shape=image_shape, name="EncoderInput")
    x = keras.layers.Conv2D(image_shape[0], 3, activation='relu', padding='same')(encoder_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(image_shape[0], 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Flatten()(x)
    encoder_output = keras.layers.Dense(encode_len, activation="relu", name="EncoderOutput")(x)
    return keras.Model(encoder_input, encoder_output, name='EncoderCNN')


def DecoderCNN(image_shape, encode_len):
    helper_shape = utils.get_helper_shape(image_shape)
    decoder_input = keras.Input(shape=(encode_len,), name="DecoderInput")
    x = keras.layers.Dense(utils.number_of_pixels(helper_shape), activation="relu")(decoder_input)
    x = keras.layers.Reshape(helper_shape)(x)
    x = keras.layers.Conv2D(image_shape[0], 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(image_shape[0], 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
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
    x = keras.layers.Dense(utils.number_of_pixels(image_shape), activation="relu", name="DecoderOutput")(x)
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