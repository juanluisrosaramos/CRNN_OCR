import argparse
from os import listdir
import os
from os.path import isfile, join, basename
from time import time
import tensorflow as tf
import numpy as np
from utils import TextFeatureIO, load_and_resize_image
from crnn_model import CRNN

"""
Script to test predictions grouping images in batches
"""
BATCH_SIZE = 32
NUMBER_OF_PREDICTIONS = 10
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)
def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', type=str, help='Where you store images', default='data/test_images')
    parser.add_argument('-w', '--weights_path', type=str, help='Where you store the weights', default='model/crnn_dsc_2018-08-20.ckpt')
    parser.add_argument('-o', '--output_file', type=str, help='Name of the csv file with the results', default='data/output.csv')
    return parser.parse_args()

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def load_images(onlyfiles, files_limit: int):
    return np.array([load_and_resize_image(p) for p in onlyfiles]), onlyfiles


def recognize(image_path: str, weights_path: str, output_file: str, files_limit=32):
    decoder = TextFeatureIO().reader
    #Read all the files in the images folder
    files = [join(image_path, f) for f in listdir(image_path) if isfile(join(image_path, f))][:]
    tf.reset_default_graph()
    inputdata = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 32, 100, 3], name='input')
    images_sh = tf.cast(x=inputdata, dtype=tf.float32)
    # build shadownet
    net = CRNN(phase='Test', hidden_nums=256, seq_length=25, num_classes=37)
    with tf.variable_scope('shadow'):
        net_out = net.build(inputdata=images_sh)
    #top_paths=NUMBER_OF_PREDICTIONS is the number of words to predict
    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(net_out, 25 * np.ones(BATCH_SIZE), merge_repeated=False,top_paths=NUMBER_OF_PREDICTIONS)

    # config tf saver
    saver = tf.train.Saver()
    sess = tf.Session()
    with sess.as_default():

        # restore the model weights
        #print('TFVERSION',tf.__version__)
        print("Restoring trained model")
        saver.restore(sess=sess, save_path=weights_path)
        print("Predicting {} images in chunks of {}".format(len(files),BATCH_SIZE))
        starting_time = time()

        #Run inference in groups of BATCH_SIZE images
        #Run it with all the files from the provided folder
        for group in chunker(files, BATCH_SIZE):
            start_time = time()
            images, filenames = load_images(group, files_limit)
            images = np.squeeze(images)
            padded_images = np.zeros([BATCH_SIZE, 32, 100, 3])
            padded_images[:images.shape[0], :, :, :] = images

            predictions,probs = sess.run([decoded,log_probabilities], feed_dict={inputdata: padded_images})
            for i, fname in enumerate(filenames):
                result = ''
                #log_probabilities is recomputed for softmax probs
                e_x = np.exp(probs[i,:]) / np.sum(np.exp(probs[i,:]))

                #build the array of N predictions for each image
                for x in range(NUMBER_OF_PREDICTIONS):
                    preds_res2 = decoder.sparse_tensor_to_str(predictions[x])
                    result = result + ',{:s},{:f}'.format(preds_res2[i],e_x[x])
                #output string formatting and writing to csv file
                result = (basename(fname) + result)
                with open(output_file, 'a') as f:
                    f.write(result)
                    f.write('\n')
            end_time = time()
            print("Prediction time for {} images: {}".format(BATCH_SIZE,end_time - start_time))

        print("Total prediction time: {}".format(end_time - starting_time))
        print("Predictions saved in file {}".format(output_file))

if __name__ == '__main__':
    params = parse_params()
    recognize(params.image_dir, params.weights_path, params.output_file)
