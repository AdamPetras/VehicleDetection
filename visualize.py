import numpy as np
import cv2
import sys
import time
import tensorflow.contrib.tensorrt as trt
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-m", "--model",
                  action="store", type="string", dest="model_path", help="Path to frozen_inference_graph.pb", default="ssd_mobilenet_v2/quantized/frozen_inference_graph.pb")      
parser.add_option("-v", "--video",
                  action="store", type="string", dest="video_path", help="Path to video", default="testing/test1.mp4")        
parser.add_option("-l", "--label",
                  action="store", type="string", dest="label_path", help="Path to label_map.pbtxt", default="Cod20k/annotations/label_map.pbtxt")   
parser.add_option("-t", "--type",
                  action="store", type="string", dest="od_type", help="Type of object detection two values supported [ssd , faster_rcnn]", default="ssd")
(options, args) = parser.parse_args()

OpenCv_window_name = 'VehicleDetection'
Threshold = 0.3
Box_color = (0,0,255)
Label_color=(255, 158, 36 )

LABELPATH = options.label_path
FROZENGRAPHPATH = options.model_path
VIDEOPATH = options.video_path
OD_TYPE = options.od_type


sys.path.append("/home/adam/Tensorflow/models/research/object_detection")
from object_detection.utils import label_map_util
import tensorflow as tf



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

def load_frozen_graph(pb_path):
    frozen_graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_path, 'rb') as pf:
        frozen_graph_def.ParseFromString(pf.read())
    for node in frozen_graph_def.node: #OPTIMALIZACE přidá 4 fps větev NONMAXSUPPRESSION poběži na CPU a rfcn poběží na GPU
        if 'NonMaxSuppression' in node.name:
            node.device = '/device:CPU:0'
        if 'rfcn_' in pb_path and 'SecondStage' in node.name:
            node.device = '/device:GPU:0'
    with tf.Graph().as_default() as frozen_graph:
        tf.import_graph_def(frozen_graph_def, name='')
    return frozen_graph



def pre_process(src, shape=None, to_rgb=True):
    img = src.astype(np.uint8)
    if shape:
        img = cv2.resize(img, shape)
    img = img[..., ::-1] # BGR na RGB
    return img

def get_boxes_with_score_and_class(img, boxes, scores, classes, threshold):
    h, w, _ = img.shape
    out_box = boxes[0] * np.array([h, w, h, w])
    out_box = out_box.astype(np.int32)
    out_score = scores[0]
    out_class = classes[0].astype(np.int32)
    mask = np.where(out_score >= threshold)    # vrácení boxů které mají větší nebo roven threshold
    return (out_box[mask], out_score[mask], out_class[mask])


def detect(origimg, tf_sess, threshold, od_type='ssd'):
    tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
    tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    if od_type == 'faster_rcnn': #pokud je faster rcnn resizenu obrázky na 1024 x 576
        img = pre_process(origimg, (1024, 576))
    elif od_type == 'ssd': #pokud je SSD tak resizenu obrázky na 300x300
        img = pre_process(origimg, (300, 300))
    else:
        raise ValueError('Wrong object detector type: '+ od_type)
    boxes_out, scores_out, classes_out = tf_sess.run(
        [tf_boxes, tf_scores, tf_classes],
        feed_dict={tf_input: img[None, ...]})
    box, score, detect_class = get_boxes_with_score_and_class(
        origimg, boxes_out, scores_out, classes_out, threshold)
    return (box, score, detect_class)

def draw_fps(img, fps):
    ms = 0
    if fps != 0:
        ms = 1./fps*1000
    fps_text = 'FPS: {:.1f} ms: {:.1f}'.format(fps,ms) # ořezání fps na 1 des
    cv2.putText(img, fps_text, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, Box_color, 1, cv2.LINE_AA)
    return img

def open_display_window(width, height):
    cv2.namedWindow(OpenCv_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(OpenCv_window_name, width, height)
    cv2.moveWindow(OpenCv_window_name, 0, 0)
    cv2.setWindowTitle(OpenCv_window_name, 'Jetson TX2 vehicle detection')

def draw_boxes_and_texts(img, box, score, cls):
    for bouding_box, score, detect_class in zip(box, score, cls):
        detect_class = int(detect_class)
        y_min, x_min, y_max, x_max = bouding_box[0], bouding_box[1], bouding_box[2], bouding_box[3]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), Box_color, 2)
        class_name = cls_dict.get(detect_class, 'CLS{}'.format(detect_class))
        txt = '{0} {1:.0%}'.format(class_name, score)
        cv2.putText(img, txt, (x_min, y_min-3), cv2.FONT_HERSHEY_PLAIN, 1.0,Label_color, thickness=1, lineType=cv2.LINE_8)
    return img

cls_dict = read_label_map(LABELPATH)
trt_graph = load_frozen_graph(FROZENGRAPHPATH)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config, graph=trt_graph)

open_display_window(1280, 720) 
show_fps = True
fps1 = 0.0
t1 = time.time()
video = cv2.VideoCapture(VIDEOPATH)
while (video.isOpened()):
    if cv2.getWindowProperty(OpenCv_window_name, 0) < 0:
        break
    _, img = video.read()
    if img is not None:
        box, score, detect_class = detect(img, tf_sess, Threshold, od_type=OD_TYPE)
        img = draw_boxes_and_texts(img, box, score, detect_class)
        if show_fps:
            img = draw_fps(img, fps1)
        cv2.imshow(OpenCv_window_name, img)
        t0 = time.time()
        curr_fps = 1.0 / (t0 - t1)
        if fps1 == 0.0:
            fps1 = curr_fps
        else:
            fps1 = (fps1*0.9 + curr_fps*0.1)
        t1 = t0
    if cv2.waitKey(1) == ord('q'):
        break
tf_sess.close()
cv2.destroyAllWindows()
