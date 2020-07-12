import numpy as np
import cv2
import sys
import time
import glob
from PIL import Image 
from optparse import OptionParser
from tensorflow.python.client import timeline
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
parser = OptionParser()
parser.add_option("-m", "--model",
                  action="store", type="string", dest="model_path", help="Path to frozen_inference_graph.pb", default="ssd_mobilenet_v2/quantized/frozen_inference_graph.pb")      
parser.add_option("-l", "--label",
                  action="store", type="string", dest="label_path", help="Path to label_map.pbtxt", default="Cod20k/annotations/label_map.pbtxt")   
(options, args) = parser.parse_args()

OpenCv_window_name = 'VehicleDetection'
Threshold = 0.3

LABELPATH = options.label_path
FROZENGRAPHPATH = options.model_path

sys.path.append("/usr/local/lib/python3.7/dist-packages/tensorflow_core/models/research/object_detection")
from object_detection.utils import label_map_util




def read_label_map(path_to_labels):

    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels)
    cls_dict = {}
    for x in category_index.values():
        cls_dict[int(x['id'])] = x['name']
    num_classes = len(cls_dict) + 1
    id_class ={}
    for i in range(num_classes):
        id_class[i] =  cls_dict.get(i, 'CLS{}'.format(i))
    return id_class

def pre_process(src,to_rgb=True):
    img = src.astype(np.uint8)
    return img


def load_frozen_graph(pb_path):
    trt_graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        trt_graph_def.ParseFromString(serialized_graph)
    for node in trt_graph_def.node: #OPTIMALIZACE přidá 4 fps větev NONMAXSUPPRESSION poběži na CPU a rfcn poběží na GPU
        if 'NonMaxSuppression' in node.name:
            node.device = '/device:CPU:0'
        if 'rfcn_' in pb_path and 'SecondStage' in node.name:
            node.device = '/device:GPU:0'
    with tf.Graph().as_default() as trt_graph:
        tf.import_graph_def(trt_graph_def, name='')
    return trt_graph

def pre_process(src, shape=None, to_rgb=True):
    img = src.astype(np.uint8)
    if shape:
        img = cv2.resize(img, shape)
    img = img[..., ::-1] # BGR na RGB
    return img

def get_boxes_with_score_and_class(img, boxes, scores, classes, threshold):
    _,h, w, _ = img.shape
    out_box = boxes[0] * np.array([h, w, h, w])
    out_box = out_box.astype(np.int32)
    out_score = scores[0]
    out_class = classes[0].astype(np.int32)
    mask = np.where(out_score >= threshold)    # vrácení boxů které mají větší nebo roven threshold
    return (out_box[mask], out_score[mask], out_class[mask])


def detect(origimg, tf_sess, threshold):
    tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
    tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    
    boxes_out, scores_out, classes_out = tf_sess.run(
        [tf_boxes, tf_scores, tf_classes],
        feed_dict={tf_input: origimg}) #,options=options,run_metadata=run_metadata
    # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    # chrome_trace = fetched_timeline.generate_chrome_trace_format()
    # with open('timeline_02_step_%d.json' % i, 'w') as f:
    #     f.write(chrome_trace)
    box, score, detect_class = get_boxes_with_score_and_class(
        origimg, boxes_out, scores_out, classes_out, threshold)
    return (box, score, detect_class)

cls_dict = read_label_map(LABELPATH)
trt_graph = load_frozen_graph(FROZENGRAPHPATH)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config, graph=trt_graph)

# options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()

batch_size = 1
iterations = 100
skip_Iter = 5
file_list = sorted(glob.glob('Cod20k/images/test/*.jpg'))
photo_Size = (300,300)
image_list = []
for i in range(0,batch_size):
    img = cv2.imread(file_list[i])
    img = pre_process(img,photo_Size)
    image_list.append(img)
batch = np.stack(image_list, axis=0)
start_time = time.time()
for currIter in range(iterations):
    if currIter == skip_Iter:
        start_time = time.time()
    box, score, detect_class = detect(batch, tf_sess, Threshold)
end_time = time.time()

time_per_lp = 1000*(end_time-start_time)/(batch_size*(iterations-skip_Iter))
print("Time per image: ", time_per_lp, " ms")
print("Batch "+str(batch_size)+": FPS: ", 1000/time_per_lp)
tf_sess.close()