import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


saver = tf.train.import_meta_graph("/app/model/crnn_dsc_2018-08-20.ckpt.meta")
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

with tf.Session() as sess:
    saver.restore(sess, "/app/model/crnn_dsc_2018-08-20.ckpt")

    # Check all operations (nodes) in the graph:
    print("## All operations: ")
    for op in graph.get_operations():
        print(op.name)

    # OR check all variables in the graph:
    print("## All variables: ")
#    for v in tf.global_variables():
        #print(v.name)

    # OR check all trainable variables in the graph:
    print("## Trainable variables: ")
    #for v in tf.trainable_variables():
        #print(v.name)

    # OR save the whole graph and weights into a text file:
    log_dir = "/log_dir"
    out_file = "train.pbtxt"
    #tf.train.write_graph(input_graph_def, logdir=log_dir, name=out_file, as_text=True)

model_file = "/app/model/crnn_dsc_2018-08-20.ckpt"
reader = pywrap_tensorflow.NewCheckpointReader(model_file)
var_to_shape_map = reader.get_variable_to_shape_map()

#for key in sorted(var_to_shape_map):
    #print("tensor_name: ", key)
    #print(reader.get_tensor(key))
