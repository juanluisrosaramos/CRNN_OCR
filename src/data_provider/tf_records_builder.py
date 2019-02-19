import numpy as np
import tensorflow as tf
from tqdm import tqdm
from os.path import join, dirname, exists
from utils import FeatureIO
from logger import LogFactory
from . import ImageDescription, Lexicon


class TfRecordBuilder:
    def __init__(self, annotations_file, output_file, lexicon: Lexicon, validate_paths=True):
        self._log = LogFactory.get_logger()
        self._annotations_file = annotations_file
        self._output_file = output_file
        self._validate = validate_paths
        self._encoder = FeatureIO()
        self._lexicon = lexicon

    def process(self):
        if not exists(self._annotations_file):
            raise IOError("File {self._annotations_file} doesn't exists")
        annotations_dir = dirname(self._annotations_file)
        self._log.info("Reading annotations file...")
        with open(self._annotations_file, 'r') as anno_file:
            infos = np.array([tmp.strip().split() for tmp in anno_file.readlines()])
        self._log.info("Number of rows in annotation file: {infos.shape[0]}")
        if self._validate:
            self._validate_paths(annotations_dir, infos)

        self._log.info("Processing...")
        with tf.python_io.TFRecordWriter(self._output_file) as writer:
            for info in tqdm(infos):
                try:
                    self.process_item(writer, annotations_dir, info)
                except Exception:
                    self._log.warn("File {info[0]} skipped")

    def process_item(self, writer, annotations_dir, info):
        #info[0] name of image, info[1] label
        label = self._lexicon.get_word_by_index(int(info[1]))
        descr = ImageDescription(info[0], annotations_dir, label)
        descr.encode_label(lambda character: self._encoder.char_to_int(character))
        feat_descr = self.create_feature_example(descr)
        writer.write(feat_descr)

    @classmethod
    def create_feature_example(cls, img_descr):
        features = tf.train.Features(feature={
            'labels': FeatureIO.int64_feature(img_descr.encoded_label),
            'images': FeatureIO.bytes_feature(img_descr.image_as_bytes()),
            'imagenames': FeatureIO.bytes_feature(img_descr.name)
        })
        example = tf.train.Example(features=features)
        return example.SerializeToString()

    def _validate_paths(self, annotations_dir, infos):
        self._log.info("Check if paths are valid...")
        inv_paths = [p for p in infos[:, 0] if not exists(join(annotations_dir, p))]
        if len(inv_paths) > 0:
            print(' --- DOES NOT EXISTS --- ')
            for p in inv_paths:
                print(p)
            raise ValueError("Invalid images")
