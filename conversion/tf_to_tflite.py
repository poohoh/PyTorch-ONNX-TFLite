import os

import tensorflow as tf

saved_model_dir = '../../model/230918_lpr/TF'
tflite_save_path =  '../../model/230918_lpr/TFLite_FP16'
tflite_model_path = os.path.join(tflite_save_path, 'lpr_fp16.tflite')

os.makedirs(tflite_save_path)

# set converter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# set fp16
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# convert model
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)