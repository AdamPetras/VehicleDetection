import time
import numpy as np
from optparse import OptionParser
import onnxruntime
import glob
import cv2
cpu = False

def postprocess(result, threshold):
    #r = np.array(result)
    scores = result[0]
    boxes = result[1]
    classes = result[2]
    # indices = result[2]
    out_boxes = []
    out_classes = []
    out_scores = []
    for i in range(0,len(boxes[0])):
        if scores[0][i] > threshold:
            out_boxes.append(boxes[0][i])
            out_classes.append(classes[0][i])
            out_scores.append(scores[0][i])
    return out_boxes,out_scores,out_classes

def get_boxes_with_score_and_class(img, boxes, scores, classes, threshold):
    _,h, w, _ = img.shape
    out_box = boxes[0] * np.array([h, w, h, w])
    out_box = out_box.astype(np.int32)
    out_score = scores[0]
    out_class = classes[0].astype(np.int32)
    mask = np.where(out_score >= threshold)    # vrácení boxů které mají větší nebo roven threshold
    return (out_box[mask], out_score[mask], out_class[mask])

def detect(origimg,output_names,input_name, sess, threshold):
    
    result = sess.run(output_names,{input_name: origimg})
    # box, detect_class,score  = get_boxes_with_score_and_class(
    #     origimg, result[1], result[0], result[2], threshold)
    # return (box, score, detect_class)

so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session = onnxruntime.InferenceSession("rcnn_training/frozen_inference_graph.onnx", sess_options=so)
input_name = session.get_inputs()[0].name
output_names = []
for i in session.get_outputs():
    output_names.append(i.name)
batch_size = 1
iterations = 100
threshold = 0.4
skip_Iter = 5
file_list = sorted(glob.glob('Cod20k/images/test/*.jpg'))
photo_Size = (300,300)
image_list = []
for i in range(0,batch_size):
    img = cv2.imread(file_list[i])
    img = cv2.resize(img, photo_Size)
    image_list.append(img)
batch = np.stack(image_list, axis=0)
for i in range(0,5):
    start_time = time.time()
    for currIter in range(iterations):
        if currIter == skip_Iter:
            start_time = time.time()
        detect(batch,output_names,input_name, session, threshold)
    end_time = time.time()
    time_per_lp = 1000*(end_time-start_time)/(batch_size*(iterations-skip_Iter))
    print("Time per image: ", time_per_lp, " ms")
    print("Batch "+str(batch_size)+": FPS: ", 1000/time_per_lp)