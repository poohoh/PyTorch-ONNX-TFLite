import os

from onnx_tf.backend import prepare
import onnx

onnx_model_path = '../../model/230914_vd/ONNX/vd.onnx'
tf_model_path = '../../model/230914_vd/TF'

os.makedirs(tf_model_path, exist_ok=True)

onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)