import os

import tensorflow as tf

saved_model_dir = '../../model/230914_vd/TF'
tflite_save_path =  '../../model/230914_vd/TFLite'
tflite_model_path = os.path.join(tflite_save_path, 'vd.tflite')

os.makedirs(tflite_save_path)

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)