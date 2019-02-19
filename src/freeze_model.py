import argparse
import tensorflow as tf
import numpy as np
from time import time
from os.path import exists
from tensorflow.python.platform import gfile
from utils import load_and_resize_image

saver = tf.train.import_meta_graph('/app/model/crnn_dsc_2018-08-18.ckpt.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "./crnn_dsc_2018-08-18")

output_node_names="y_pred"
output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session
            input_graph_def, # input_graph_def is useful for retrieving the nodes
            output_node_names.split(",")  )

output_graph="/app/model/crnn_dsc_2018-08-18.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()
