import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
save_dir="./rcnn_training/"
with tf.Session() as sess:
    converter = trt.TrtGraphConverter(
    input_saved_model_dir=save_dir+"my-model/saved_model",
    max_batch_size=1,
    max_workspace_size_bytes=1 << 24,
    precision_mode='FP16',
    minimum_segment_size=5
    )
    trt_graph = converter.convert()
    with open(save_dir+"quantized/frozen_inference_graph.pb", 'wb') as f:
        f.write(trt_graph.SerializeToString())