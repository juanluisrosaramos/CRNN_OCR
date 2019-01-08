import argparse
from os import listdir

from os.path import isfile, join, basename
from time import time
import tensorflow as tf
import numpy as np
from utils import TextFeatureIO, load_and_resize_image
from crnn_model import CRNN


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', type=str, help='Where you store images', default='data/test_images')
    parser.add_argument('-w', '--weights_path', type=str, help='Where you store the weights')
    return parser.parse_args()


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def load_images(onlyfiles, files_limit: int):
    #onlyfiles = [join(image_path, f) for f in listdir(image_path) if isfile(join(image_path, f))][:files_limit]
    #for group in chunker(onlyfiles, 64):
    #    print (group)
    return np.array([load_and_resize_image(p) for p in onlyfiles]), onlyfiles


def recognize(image_path: str, weights_path: str, files_limit=32):
    decoder = TextFeatureIO().reader
    #Read all the files in the folder
    files = [join(image_path, f) for f in listdir(image_path) if isfile(join(image_path, f))][:]
    #Divide the files in chuncks of 64 files
    #for group in chunker(files, 64):
        # images, filenames = load_images(group, files_limit)
        # images = np.squeeze(images)
        # padded_images = np.zeros([64, 32, 100, 3])
        # padded_images[:images.shape[0], :, :, :] = images

    tf.reset_default_graph()

    inputdata = tf.placeholder(dtype=tf.float32, shape=[64, 32, 100, 3], name='input')

    images_sh = tf.cast(x=inputdata, dtype=tf.float32)

    # build shadownet
    net = CRNN(phase='Test', hidden_nums=256, seq_length=25, num_classes=37)
    with tf.variable_scope('shadow'):
        net_out = net.build(inputdata=images_sh)
    #top paths is the number of words to predict
    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(net_out, 25 * np.ones(64), merge_repeated=False,top_paths=10)

    # config tf saver
    saver = tf.train.Saver()
    sess = tf.Session()
    with sess.as_default():

        # restore the model weights
        print('TFVERSION',tf.__version__)
        print("Restoring trained model")
        saver.restore(sess=sess, save_path=weights_path)
        print("Predicting {} images in chuncks of 64".format(len(files)))
        starting_time = time()
        for group in chunker(files, 64):
            start_time = time()
            images, filenames = load_images(group, files_limit)
            images = np.squeeze(images)
            padded_images = np.zeros([64, 32, 100, 3])
            padded_images[:images.shape[0], :, :, :] = images

            predictions = sess.run(decoded, feed_dict={inputdata: padded_images})
            probs = sess.run(log_probabilities, feed_dict={inputdata: padded_images})

            #preds_res = decoder.sparse_tensor_to_str(predictions[0])
            for i, fname in enumerate(filenames):
                result = ''
                e_x = np.exp(probs[i,:]) / np.sum(np.exp(probs[i,:]))
                for x in range(10):
                    preds_res2 = decoder.sparse_tensor_to_str(predictions[x])
                    result = result + ',{:s},{:f}'.format(preds_res2[i],e_x[x])
                    #print("{}: {}, {:f}".format(fname, preds_res2[i],e_x[x]))
                result = (basename(fname) + result)
                with open('fileName.csv', 'a') as f:
                    f.write(result)
                    f.write('\n')
                #print('III',i,result)
            end_time = time()
            print("Prediction time for 64 images: {}".format(end_time - start_time))

        print("Total prediction time: {}".format(end_time - starting_time))

if __name__ == '__main__':
    params = parse_params()
    recognize(params.image_dir, params.weights_path)
