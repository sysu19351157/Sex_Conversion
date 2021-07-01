# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from model import CycleGAN
from imageio import imread
import os
import cv2
import pickle
from tqdm import tqdm

with open('../test.pkl', 'rb') as fr:
    pos = pickle.load(fr)

cap = cv2.VideoCapture('../test.mp4')
ret, image_np = cap.read()
out = cv2.VideoWriter('../output1.mp4', -1, cap.get(cv2.CAP_PROP_FPS), (image_np.shape[1], image_np.shape[0]))

frames = []
while cap.isOpened():
    ret, image_np = cap.read()
    if len((np.array(image_np)).shape) == 0:
        break
    frames.append(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

image_size = 256
image_file = 'face.jpg'
for gender in ['M', 'F']:
    if gender == 'M':
        model = '../pretrained/male2female.pb'
    else:
        model = '../pretrained/female2male.pb'

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(model, 'rb') as model_file:
            graph_def.ParseFromString(model_file.read())
            tf.import_graph_def(graph_def, name='')

        with tf.Session(graph=graph) as sess:
            input_tensor = graph.get_tensor_by_name('input_image:0')
            output_tensor = graph.get_tensor_by_name('output_image:0')

            for i in tqdm(range(len(pos))):
                fid, y0, y1, x0, x1, h, w, g = pos[i]
                if g == gender:
                    face = cv2.resize(frames[fid - 1][y0: y1, x0: x1, :], (image_size, image_size))
                    output_face = sess.run(output_tensor, feed_dict={input_tensor: face})

                    with open(image_file, 'wb') as f:
                        f.write(output_face)

                    output_face = imread(image_file)
                    maxv = np.max(output_face)
                    minv = np.min(output_face)
                    output_face = ((output_face - minv) / (maxv - minv) * 255).astype(np.uint8)

                    output_face = cv2.resize(output_face, (w, h))
                    frames[fid - 1][y0: y1, x0: x1, :] = output_face

for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
os.remove(image_file)
cap.release()
out.release()
cv2.destroyAllWindows()