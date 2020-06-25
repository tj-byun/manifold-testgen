import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import layers


resize_layer = tf.keras.layers.Lambda(
    lambda image: tf.image.resize(
        image,
        (96, 96),
        method=tf.image.ResizeMethod.BILINEAR,
        preserve_aspect_ratio=True
    )
)

model_path = os.path.join('models', 'lecs', 'xception.h5')
pretrained = load_model(model_path)

model = Sequential()
model.add(layers.InputLayer(input_shape=(32, 32, 3)))
model.add(resize_layer)
model.add(pretrained)
model.save(model_path + '.resized.h5')
