import numpy as np
import os
import sys
from keras.models import load_model
import cv2
from two_stage_vae import limit_keras_gpu_usage
import random

assert len(sys.argv) == 3, "$ python show_result.py <task> <dir>"
task = sys.argv[1]
dir = sys.argv[2]

limit_keras_gpu_usage(0.45)

LABELS = {
    "mnist": [str(i) for i in range(10)],
    "fashion": ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'],
    "taxinet": ['far left', 'left', 'center', 'right', 'far right'],
    "lecs": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                'horse', 'ship', 'truck'],
}
assert task in LABELS
labels = LABELS[task]
print("labels", labels)

model = load_model(os.path.join('testmodels', task + '.h5'))

x = np.load(os.path.join(dir, 'x.npy'))
y = np.load(os.path.join(dir, 'y.npy'))
print("y", y)
try:
    v = np.load(os.path.join(dir, 'v.npy'))
except:
    v = [True] * len(x)
y_pred = model._get_discriminator(x / 255.)

assert len(x) == len(y) == len(v) == len(y_pred), "Lengths do not match"
print("x.shape", x.shape)


total_cnt, cnt, invalid_cnt = 1000, 0, 0
selected = []
indices = list(range(len(x)))
random.shuffle(indices)

for ind in indices:
    if not v[ind]:
        continue
    # img = cv2.cvtColor(x[i], cv2.COLOR_BGR2RGB)
    # font = ImageFont.truetype("LinLibertine_R.otf")
    pred = np.argmax(y_pred[ind])
    truth = y[ind]
    p = y_pred[ind][pred] * 100
    print("{}, truth: {}, pred : {}, ({:.1f})".format(cnt, labels[truth],
                                                      labels[pred], p))
    if pred == truth:
        print("  INVALID: Prediction and label shouldn't match!!")
        invalid_cnt += 1
    filename = os.path.join(dir, '{}.png'.format(cnt))
    cv2.imwrite(filename, x[ind])
    cnt += 1
    if cnt >= total_cnt:
        break

print("{} Invalid cases".format(invalid_cnt))
