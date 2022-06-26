# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
# from sort import *
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import argparse
import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
import time
from shapely.geometry import box
# import car
import servo
import math
import collections
from flask import Flask, render_template,request, Markup, Response

# creating object module to store results obtained from model as dictionary
# id is class id, score is probability, and bbox
# object is already obtained fron pycoral library, this second object is used
# so we only have values for cars and persons
#
# Object2 = collections.namedtuple('Object', ['id', 'score', 'bbox'])

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)


#from sort import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bc3a3cbf29c31d9b32690d356360d075'

# initialize parameters
mot_tracker = Sort() 
start = time.time()
centroid = [320,240]
idx = 0
track_id = 0

def main():
    default_model_dir = './'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_idx)
    global count
    global center_points_prev_frame
    center_points_prev_frame = []
    count = 0
    

    while cap.isOpened():
        ret, frame = cap.read()
        count += 1
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())  # detect objects in frame

        objs = get_objects(interpreter, args.threshold)[:args.top_k]  # now obtain all results

        if len(objs) != 0:
            #print('objs',objs)
            cv2_im = append_objs_to_img(frame, cv2_im, inference_size, objs, labels)  # if any object is added pass it to func to draw results on frame

        # passing resulted frame to web app
        try:
            ret, buffer = cv2.imencode('.jpg', cv2_im)
            cv2_im = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + cv2_im + b'\r\n')
        except Exception as e:
            pass


def append_objs_to_img(img, cv2_im, inference_size, objs, labels):
    global centroid, scale_x, scale_y, track_id
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    # track_id = np.arange(len(objs))
    # center_points_cur_frame = []
    
    # tracking_objects = {}

    if objs:
        # try:
        #     centroid = center_coord(objs,idx)
        #     #print('obj',centroid, 'ID)
        #     reset_servo(centroid)
        # except Exception as e:
        #     # centroid = center_coord(objs,0)
        #     # reset_servo(centroid)    # else:
        #     reset_servo([320,240])

        pop_objs = []  # to store only person and car results

        for e,obj in enumerate(objs):
            if obj.id != 43 | 2:
                # new_obj = Object2(id=obj.id,score=obj.score,
                #         bbox=obj.bbox)
                pop_objs.append(obj)

        for pop in pop_objs:
            objs.remove(pop)

        if objs:

            
            detection=[]  # storing bounding box and probab. Based on probab id is assigned by sort method
            bboxes = []
            scores = []
            names = []
            classes=[]


            for n_obj in objs:        
                bbox = n_obj.bbox.scale(scale_x, scale_y)
                x_min, y_min = int(bbox.xmin), int(bbox.ymin)
                x_max, y_max = int(bbox.xmax), int(bbox.ymax)

                element = []
                element.append(x_min)
                element.append(y_min)
                element.append(x_max)
                element.append(y_max)

                bboxes.append(element)
                score.append(obj.score)
                classes.append(n_obj.id)
                names.append(labels.get(n_obj.id))

                # element.append(score)
                # detection.append(element)
                # detection_label.append(n_obj.id)
            features = encoder(img, bboxes)
            names = np.array(names)
            scores = np.array(scores)
            detections = [Detection(bbox, scores, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            tracker.predict()
            tracker.update(detections)

            # detection = np.array(detection)
            # tr_update = mot_tracker.update(detection)  # now assigning id through sort

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update >1:
                    continue
                bbox = track.to_tlbr()
                class_name= track.get_class()
                label = 'ID {},  {}'.format(track.track_id, class_name)
                x0 = bbox[0]
                y0 = bbox[1]
                x1 = bbox[2]
                y1 = bbox[3]

                cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)


            # try:
            #     centroid = center_coord(tr_update,idx)  
            #     #print('obj',centroid, 'ID)
            #     # print('centre',centroid)
            #     reset_servo(centroid)
            # except:
            #     temp_idx = tr_update[0]
            #     temp_idx = int(temp_idx[4])
            #     centroid = center_coord(tr_update2,temp_idx)
            #     reset_servo(centroid)
            #     # pass
            #     # centroid = center_coord(objs,0)
            #     # reset_servo(centroid)    # else:
            #     # reset_servo([320,240])

            # for e,data in enumerate(tr_update):
            #     x0, y0, x1, y1, track_id = data
            #     x0, y0 = int(x0), int(y0)
            #     x1, y1 = int(x1), int(y1)
            #     track_id = int(track_id)

            #     polygon=box(*(x0,y0,x1,y1))
            #     cx, cy = [int(polygon.centroid.x), int(polygon.centroid.y)]
            #     # center_points_cur_frame.append((cx, cy)) 
            #     # print(x0,y0,x1,y1, height, width, scale_x, scale_y, inference_size[0], inference_size[1],obj.id)

            #     # percent = int(100 * obj.score)
            #     # label = '{}% {}'.format(percent, labels.get(obj.id))
            #     label = 'ID {}'.format(track_id)#, labels.get(detection_label[e]))

            #     cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            #     cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
            #                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)



            # if len(new_objs) != 0:
            #     detection=[]  # storing bounding box and probab. Based on probab id is assigned by sort method
            #     detection_label=[]
            #     for obj in new_objs:
            #         bbox = obj.bbox.scale(scale_x, scale_y)
            #         x_min, y_min = int(bbox.xmin), int(bbox.ymin)
            #         x_max, y_max = int(bbox.xmax), int(bbox.ymax)
            #         score = obj.score

            #         element = []
            #         element.append(x_min)
            #         element.append(y_min)
            #         element.append(x_max)
            #         element.append(y_max)
            #         element.append(score)
            #         detection.append(element)
            #         detection_label.append(obj.id)


            #     detection = np.array(detection)
            #     tr_update = mot_tracker.update(detection)  # now assigning id through sort
            #     tr_update2 = tr_update.copy()
            #     print('sort result:' tr_update)
            #     # centroid = center_coord(tr_update,idx)  

            #     try:
            #         centroid = center_coord(tr_update,idx)  
            #         #print('obj',centroid, 'ID)
            #         print('centre',centroid)
            #         reset_servo(centroid)
            #     except:
            #         temp_idx = tr_update2[0]
            #         temp_idx = int(temp_idx[4])
            #         centroid = center_coord(tr_update2,temp_idx)
            #         reset_servo(centroid)
            #         # pass
            #         # centroid = center_coord(objs,0)
            #         # reset_servo(centroid)    # else:
            #         # reset_servo([320,240])

            #     for e,data in enumerate(tr_update):
            #         x0, y0, x1, y1, track_id = data
            #         x0, y0 = int(x0), int(y0)
            #         x1, y1 = int(x1), int(y1)
            #         track_id = int(track_id)

            #         polygon=box(*(x0,y0,x1,y1))
            #         cx, cy = [int(polygon.centroid.x), int(polygon.centroid.y)]
            #         center_points_cur_frame.append((cx, cy)) 
            #         # print(x0,y0,x1,y1, height, width, scale_x, scale_y, inference_size[0], inference_size[1],obj.id)

            #         # percent = int(100 * obj.score)
            #         # label = '{}% {}'.format(percent, labels.get(obj.id))
            #         label = 'ID {},  {}'.format(track_id, labels.get(detection_label[e]))

            #         cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            #         cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
            #                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                    
    return cv2_im

def center_coord(detection_data,idx):
    # the id index obtained from user is searched in the sort result list
    # when id matches the row element of idx, then that row index is returned as target index
    # this target index bbox is used to obtain the centroid of detection
    # when this centroid is obtained it is passed to servo 
    #
    target_index = [e for e,i in enumerate(detection_data) if int(i[4])==idx]
    print('target',target_index)
    if target_index:
        print('target:', target_index, 'idx: ',idx)
        target_index = target_index[0]
        boxes = detection_data[target_index]
        xmin = int(boxes[0])
        ymin = int(boxes[1])
        xmax = int(boxes[2])
        ymax = int(boxes[3])
        # print('got coord')
        # bbox = boxes.bbox.scale(scale_x, scale_y)
        # xmin, ymin, xmax, ymax = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
        bounds = (xmin,ymin,xmax,ymax)
        polygon=box(*bounds)
        
        centroid = [polygon.centroid.x, polygon.centroid.y]
    else:
        boxes = detection_data[0]
        xmin = int(boxes[0])
        ymin = int(boxes[1])
        xmax = int(boxes[2])
        ymax = int(boxes[3])

        bounds = (xmin,ymin,xmax,ymax)
        polygon=box(*bounds)    
        centroid = [polygon.centroid.x, polygon.centroid.y]    
    return centroid

def reset_servo(centroid):
    servo.servo_movement(centroid)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/video_feed')
def video_feed():
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    # this function control the input obtained from user through web app
    global idx
    if request.method == 'POST':
        if request.form.get('Reset') == 'Reset':
            print(1)
        elif request.form.get('car') == 'Car Control':
            # car.car_control()
            print('car')
        elif request.form.get('Target') == 'Target':
            idx = request.form['text']
                          
                 
    elif request.method=='GET':
        return render_template('home.html')
    return render_template('home.html')

if __name__ == '__main__':
    app.run(host = '192.168.43.5',port=5000,use_reloader=True, debug=True)
