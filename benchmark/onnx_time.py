import argparse
import time
import onnxruntime as ort
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='onnx model benchmarking')
    parser.add_argument('--onnx', default='model.onnx', type=str, required=True)
    parser.add_argument('--device', default='GPU', type=str, choices=['CPU', 'GPU'])
    parser.add_argument('--batch', default=1, type=int)

    return parser.parse_args()

def main(args):
    if args.device == 'CPU':
        session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    else:
        session = ort.InferenceSession(args.onnx, providers=["CUDAExecutionProvider"])

    # run model
    sum = 0
    for i in range(10000):

        # input data
        input_data = np.random.randn(args.batch, 3, 416, 416).astype(np.float32)
        input_name = session.get_inputs()[0].name

        start = time.time()
        results = session.run(None, {input_name: input_data})
        end = time.time()
        sum += end - start

        if i % 100 == 0:
            print(f'i: {i}, sum: {sum}')
    avg_elapsed = sum / 10000

    # result
    print(results)
    print(f'elapsed time: {avg_elapsed}')

if __name__ == '__main__':
    args = get_parser()
    main(args)