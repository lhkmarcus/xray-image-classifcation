import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers

### VGG16 Model Architecture ###
## CNN BLock class for basic convolutional layer
class cnnBlock(layers.Layer):
    def __init__(self, output_channels, kernel_size=3):
        super().__init__()
        self.conv = layers.Conv2D(
            output_channels,
            kernel_size,
            padding="same"
        )
        self.bn = layers.BatchNormalization()
    
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)

## VGG BLock class built on multiple CNN Blocks
class vggBlock(layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = cnnBlock(output_channels=channels[0])
        self.conv2 = cnnBlock(output_channels=channels[1])
        self.pool = layers.MaxPooling2D()
        if len(channels) == 2: 
            self.conv3 = None
        if len(channels) == 3: 
            self.conv3 = cnnBlock(output_channels=channels[2])

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        if self.conv3 == None:
            return self.pool(x)
        else:
            x = self.conv3(x)
            return self.pool(x)

## VGG Model class built on multiple VGG Blocks
class vggModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.scale = layers.Rescaling(1./255)
        self.cnn1 = vggBlock([ 64,  64])
        self.cnn2 = vggBlock([128, 128])
        self.cnn3 = vggBlock([256, 256, 256])
        self.cnn4 = vggBlock([512, 512, 512])
        self.cnn5 = vggBlock([512, 512, 512])
        self.pool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(4096)
        self.bn = layers.BatchNormalization()
        self.classifier = layers.Dense(3, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=False):
        x = self.scale(inputs)
        x = self.cnn1(x, training=training)
        x = self.cnn2(x, training=training)
        x = self.cnn3(x, training=training)
        x = self.cnn4(x, training=training)
        x = self.cnn5(x, training=training)
        x = self.pool(x)
        x = self.dense(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return self.classifier(x)

    ## Graph function to overwrite call method for plotting output shapes
    def graph(self):
        inputs = tf.keras.Input(shape=(300, 300, 1))
        return keras.Model(inputs, self.call(inputs), name="vgg16")