import argparse

import numpy as np
import tensorflow as tf

import cv2

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef() # replaced graph_def = tf.GraphDef() for version compatiblity

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_video_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    
    if file_name.endswith(".mp4"):
        cap = cv2.VideoCapture(file_name)
    else:
        raise Exception("The input file must be in mp4 format")
    sess = tf.compat.v1.Session()
    frames = []
    read = True
    while read:
      read, img = cap.read()
      if read:
        frame = tf.convert_to_tensor(img)
        float_caster = tf.cast(frame, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        frames.append(sess.run(normalized))
    return frames

def load_labels(label_file):
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    return [l.rstrip() for l in proto_as_ascii_lines]

if __name__ == "__main__":
    file_name = "data/ankb2.mp4"
    model_file = \
    "model/inception_v3_2016_08_28_frozen.pb"
    label_file = "model/imagenet_slim_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "input"
    output_layer = "InceptionV3/Predictions/Reshape_1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="video to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.video:
        file_name = args.video
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    graph = load_graph(model_file)
    tf.compat.v1.disable_eager_execution()
    ts = read_tensor_from_video_file(
        file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    objs = []
    for fn,t in enumerate(ts):
        with tf.compat.v1.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        top_k = results.argsort()[-3:][::-1]
        labels = load_labels(label_file)
        temp = []
        for i in top_k:
            temp.append([labels[i],results[i]])
            # if results[i] > 0.6:
            #     print(f"frame number {fn}", labels[i], results[i])
        objs.append(temp)

from vid_utils import play_vid

play_vid(file_name,objs)