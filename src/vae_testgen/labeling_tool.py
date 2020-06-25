import cv2
import os
import argparse
import numpy as np
# https://subscription.packtpub.com/book/application_development/9781788474443
# /1/ch01lvl1sec18/handling-user-input-from-a-keyboard

labels = {
    "mnist": [str(i) for i in range(10)],
    "fashion": ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'],
    "taxinet": ['far left', 'left', 'center', 'right', 'far right'],
}


def main(args):
    testset_dir = args.testset_dir
    assert os.path.exists(testset_dir) and os.path.isdir(testset_dir)
    xs = np.load(os.path.join(testset_dir, 'x.npy'))
    ys = np.load(os.path.join(testset_dir, 'y.npy'))
    global labels
    labels = labels[args.task]

    v = np.array([False] * len(xs))
    print("xs.shape", xs.shape)
    quit = False
    i = 0
    while i < len(xs):
        print("Image {}/{}: {}".format(i, len(xs), labels[ys[i]]))
        cv2.imshow("result", xs[i] / 255.)
        key = cv2.waitKey(0)
        if key == ord('v') or key == ord('o') or key == ord('1') or key == \
                ord('y'):
            # valid
            v[i] = True
        elif key == ord('q'):
            quit = True
        elif key == ord('x') or key == ord('0'):
            # invalid
            v[i] = False
        elif key == ord('p'):
            i = max(i - 2, 0)
        i += 1
        if quit:
            return
    np.save(os.path.join(testset_dir, "v.npy"), v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('testset_dir', type=str)
    _args = parser.parse_args()
    # limit_keras_gpu_usage(0.3)
    main(_args)

