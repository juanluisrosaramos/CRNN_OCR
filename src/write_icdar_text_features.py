import argparse
import os
import os.path as ops
from data_provider import IcdarTfRecordsBuilder


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, help='Where you store the dataset')
    parser.add_argument('-s', '--save_dir', type=str, help='Where you store tfrecords')
    return parser.parse_args()


def write_features(dataset_dir, save_dir):
    if not ops.exists(save_dir):
        os.makedirs(save_dir)

    # write test tfrecord
    print('Start writing testing tf records')
    test_builder = IcdarTfRecordsBuilder(dataset_dir, ops.join(save_dir, 'test_feature.tfrecords'))
    test_builder.process()


if __name__ == '__main__':
    params = parse_params()
    dataset_dir = params.dataset_dir
    save_dir = params.save_dir
    if not ops.exists(dataset_dir):
        raise ValueError("Dataset {} doesn't exist".format(dataset_dir))
    write_features(dataset_dir=dataset_dir, save_dir=save_dir)
