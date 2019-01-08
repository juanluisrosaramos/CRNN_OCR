import os
import os.path as ops
import argparse
from data_provider import TfRecordBuilder, Lexicon


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, help='Where you store the dataset')
    parser.add_argument('-s', '--save_dir', type=str, help='Where you store tfrecords')
    return parser.parse_args()


def write_features(dataset_dir, save_dir):
    if not ops.exists(save_dir):
        os.makedirs(save_dir)

    lexicon = Lexicon(ops.join(dataset_dir, 'lexicon.txt'))

    # write val tfrecords
    print('Start writing validation tf records')
    val_builder = TfRecordBuilder(ops.join(dataset_dir, 'annotation_val.txt'), ops.join(save_dir, 'validation_feature.tfrecords'), lexicon)
    val_builder.process()

    # write test tfrecord
    print('Start writing testing tf records')
    test_builder = TfRecordBuilder(ops.join(dataset_dir, 'annotation_test.txt'), ops.join(save_dir, 'test_feature.tfrecords'), lexicon)
    test_builder.process()

    # write train tfrecords
    print('Start writing training tf records')
    train_builder = TfRecordBuilder(ops.join(dataset_dir, 'annotation_train.txt'), ops.join(save_dir, 'train_feature.tfrecords'), lexicon)
    train_builder.process()


if __name__ == '__main__':
    params = parse_params()
    dataset_dir = params.dataset_dir
    save_dir = params.save_dir
    if not ops.exists(dataset_dir):
        raise ValueError("Dataset {} doesn't exist".format(dataset_dir))
    write_features(dataset_dir=dataset_dir, save_dir=save_dir)
