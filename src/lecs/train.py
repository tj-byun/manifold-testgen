import tensorflow as tf
import tensorflow.keras as keras
import os
import numpy as np
import argparse

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import vgg16, vgg19, mobilenet_v2, \
    resnet_v2, xception, densenet
from tensorflow.keras.experimental import CosineDecay

from two_stage_vae import dataset
from utility import limit_keras_gpu_usage

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help="Dataset")
parser.add_argument('filename', type=str, help="Name of the model file")
parser.add_argument('--arch', type=str, default='vgg16', help="Architecture")
parser.add_argument('--epochs', type=int, default=100, help="Epochs")
parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--drop', type=int,
                    help="Drop a specified class in training dataset")
parser.add_argument("--limit-gpu", type=float, default=0.7,
                    help='Limit GPU mem usage by percentage 0 < f <= 1')
args = parser.parse_args()

limit_keras_gpu_usage(args.limit_gpu)
dir = os.path.join('models', args.dataset)
if not os.path.exists(dir):
    os.mkdir(dir)
model_path = os.path.join(dir, '{}.h5'.format(args.filename))

x_train, input_shape = dataset.load_dataset(args.dataset, 'train',
                                            normalize=True)
y_train, n_class = dataset.load_dataset(args.dataset, 'train', label=True)
if args.drop:
    x_train, y_train = dataset.drop_class(x_train, y_train, args.drop)
x_val, _ = dataset.load_dataset(args.dataset, 'test', normalize=True)
y_val, _ = dataset.load_dataset(args.dataset, 'test', label=True)

# model = efn.EfficientNetB0(classes=n_class)
if args.arch == 'xception':
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Lambda(
        lambda image: tf.image.resize(
            image,
            (96, 96),
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=True
        )
    ))
    model.add(xception.Xception(
        weights=None,
        include_top=True,
        classes=n_class,
        input_shape=(96, 96, 3)
    ))
else:
    if args.arch == 'vgg19':
        arch = vgg19.VGG19
    elif args.arch == 'vgg16':
        arch = vgg16.VGG16
    elif args.arch == 'mobilenetv2':
        arch = mobilenet_v2.MobileNetV2
    elif args.arch == 'resnet2':
        arch = resnet_v2.ResNet50V2
    elif args.arch == 'densenet':
        arch = densenet.DenseNet121
    else:
        raise Exception("Unsupported architecture type")

    model = keras.models.Sequential()
    if x_train.shape[1] < 32:
        model.add(keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Lambda(
            lambda image: tf.image.resize(
                image,
                (32, 32),
                method=tf.image.ResizeMethod.BILINEAR,
                preserve_aspect_ratio=True
            )
        ))
        input_shape = (32, 32, input_shape[2])
    model.add(arch(weights=None, classes=n_class, input_shape=input_shape,
                   include_top=False))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(2 ** (n_class - 1), activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(2 ** (n_class - 2), activation='relu'))
    model.add(keras.layers.Dense(n_class, activation='softmax'))

model.summary()

optim = Adam(lr=args.lr)
model.compile(loss='categorical_crossentropy', optimizer=optim,
              metrics=['accuracy'])

# callbacks
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
lrate = LearningRateScheduler(CosineDecay(0.0001, args.epochs))
callbacks = [checkpoint, lrate]

model.fit(x_train, to_categorical(y_train, n_class),
          batch_size=args.batch_size,
          epochs=args.epochs,
          validation_data=(x_val, to_categorical(y_val, n_class)),
          callbacks=callbacks)

