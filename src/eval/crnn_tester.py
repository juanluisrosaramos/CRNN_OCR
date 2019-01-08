import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from utils import TextFeatureIO
from logger import LogFactory
from config import GlobalConfig
from crnn_model import CRNN


class CrnnTester(ABC):
    def __init__(self, tfrecords_path: str, weights_path: str, config: GlobalConfig):
        self._log = LogFactory.get_logger()
        self._tfrecords_path = tfrecords_path
        self._weights_path = weights_path
        self._batch_size = config.get_test_config().batch_size
        self._merge_repeated = config.get_test_config().merge_repeated_chars
        self._gpu_config = config.get_gpu_config()
        self._decoder = TextFeatureIO().reader
        self._recognition_time = None

    def run(self):
        self._recognition_time = []
        images_sh, labels_sh, imagenames_sh = self.load_data()
        images_sh = tf.cast(x=images_sh, dtype=tf.float32)

        net = CRNN(phase='Test', hidden_nums=256, seq_length=25, num_classes=37)
        with tf.variable_scope('shadow'):
            net_out = net.build(inputdata=images_sh)
        decoded, _ = tf.nn.ctc_beam_search_decoder(net_out, 25 * np.ones(self._batch_size), merge_repeated=self._merge_repeated)
        sess_config = self.config_tf_session()

        # config tf saver
        saver = tf.train.Saver()
        sess = tf.Session(config=sess_config)

        with sess.as_default():
            # restore the model weights
            saver.restore(sess=sess, save_path=self._weights_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            self._log.info('Start predicting ...')
            accuracy, distance = self.test(decoded, imagenames_sh, images_sh, labels_sh, sess)
            coord.request_stop()
            coord.join(threads=threads)
        sess.close()
        avg_time = np.mean(self._recognition_time)
        return accuracy, distance, avg_time

    def config_tf_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = self._gpu_config.memory_fraction
        sess_config.gpu_options.allow_growth = self._gpu_config.is_tf_growth_allowed()
        return sess_config

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def test(self, decoded, imagenames_sh, images_sh, labels_sh, sess):
        pass
