import argparse
import numpy as np
import tensorflow as tf
import time

def get_parser():
    parser = argparse.ArgumentParser(description='Run TF Model')
    parser.add_argument('--tflite_model', default='model', type=str, required=True)
    parser.add_argument('--device', default='GPU', type=str, choices=['CPU', 'GPU'])
    parser.add_argument('--batch', default=1, type=int)

    return parser.parse_args()

def main(args):
    # Load the TFLite model and allocate tensors
    if args.device == 'GPU':
        interpreter = tf.lite.Interpreter(
            model_path=args.tflite_model,
            experimental_delegates=[tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.so')]
        )
        print('GPU available')
    else:
        interpreter = tf.lite.Interpreter(model_path=args.tflite_model, num_threads=4)
    
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    sum = 0
    for i in range(1000):
        # set input image
        input_data = np.random.randn(args.batch, 3, 416, 416).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        start = time.time()
        interpreter.invoke()
        end = time.time()

        sum += end - start
    
        if i % 100 == 0:
            print(f'i: {i}, sum: {sum}')
    
    avg_elapsed = sum / 1000
    print(f'elapsed time: {avg_elapsed}')

if __name__=='__main__':
    args = get_parser()
    main(args)