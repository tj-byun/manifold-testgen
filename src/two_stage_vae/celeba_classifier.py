import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2
from two_stage_vae.dataset import load_dataset, load_attribute
from utility import limit_keras_gpu_usage


def get_model(input_shape, n_outputs):
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), kernel_regularizer=l2(0.0001),
                            input_shape=input_shape, padding='same'))
    model.add(layers.Conv2D(32, (5, 5), (2, 2), kernel_regularizer=l2(0.0001),
                            padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(48, (3, 3), kernel_regularizer=l2(0.0001),
                            padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D(padding='same'))
    model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=l2(0.0001),
                            padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64))
    model.add(layers.Dense(32))
    model.add(layers.Dense(n_outputs, activation='sigmoid'))
    # print("output shape: ", model.output.shape, 'output0', model.output[:, 0])
    return model


def get_mae(index):
    def f(y_true, y_pred):
        return keras.losses.mae(y_true[:, index], y_pred[:, index])

    return f


def main():
    batch_size = 64
    xs, n_channel = load_dataset('celeba', normalize=True)
    attrs, labels = load_attribute('celeba', category='train')
    assert len(xs) == len(attrs), \
        "len(xs): {}, len(attrs): {}".format(len(xs), len(attrs))

    model = get_model(xs.shape[-3:], len(labels))
    model.compile(keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=[get_mae(i) for i in range(len(labels))]
                  )
    model.summary()
    print('xs', xs.shape, 'ys', attrs.shape)
    model.fit(x=xs, y=attrs, batch_size=batch_size)


if __name__ == '__main__':
    limit_keras_gpu_usage(0.5)
    main()
