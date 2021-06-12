from tensorflow import keras
import utils


class CNNBlock(keras.layers.Layer):
    def __init__(self, out_channels, down_sample, pool=0, activation='relu', kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv = keras.layers.Conv2D(out_channels, kernel_size, activation=activation, padding="same")
        self.bn = keras.layers.BatchNormalization()
        if pool > 1:
            self.pool = True
            if not down_sample:
                self.pooling = keras.layers.UpSampling2D(pool)
            else:
                self.pooling = keras.layers.MaxPool2D(pool)
        else:
            self.pool = False


    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        if self.pool:
            x = self.pooling(x, training=training)
        return x


class DecoderOutput(keras.layers.Layer):
    def __init__(self, image_shape, activation='relu'):
        super(DecoderOutput, self).__init__()
        self.d = keras.layers.Dense(utils.number_of_pixels(image_shape), activation=activation)
        self.r = keras.layers.Reshape(image_shape)


    def call(self, input_tensor, training=False):
        x = self.d(input_tensor)
        x = self.r(x, training=training)
        return x