import argparse
import time

import tensorflow as tf
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='elapsed time TF Model')
    parser.add_argument('--tf_model', default='model', type=str, required=True)
    parser.add_argument('--device', default='GPU', type=str, choices=['CPU', 'GPU'])
    parser.add_argument('--batch', default=1, type=int)


    return parser.parse_args()


def main(args):
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("GPU available")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("GPU not available")

    if args.device == 'CPU':
        device_name = '/CPU:0'
    else:
        device_name = '/GPU:0'

    with tf.device(device_name):
        model = tf.saved_model.load(args.tf_model)
        model.trainable = False

        # warm up
        for i in range(10):
            input_data = np.random.randn(args.batch, 3, 416, 416).astype(np.float32)
            model(**{'input': input_data})

        sum = 0
        for i in range(1000):
            input_data = np.random.randn(args.batch, 3, 416, 416).astype(np.float32)

            start = time.time()
            out = model(**{'input': input_data})
            end = time.time()

            sum += end - start

            if i % 100 == 0:
                print(f'i: {i}, sum: {sum}')

        avg_elapsed = sum / 1000

        print(out)
        print(f'elapsed time: {avg_elapsed}')

if __name__=='__main__':
    args = get_parser()
    main(args)