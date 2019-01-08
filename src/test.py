import os.path as ops
import argparse
from config import ConfigProvider, GlobalConfig
from logger import LogFactory
from eval import RecursiveCrnnTester, BasicCrnnTester


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, metavar='PATH', help='Path to config file')
    parser.add_argument('-d', '--dataset_dir', type=str, help='Where you store the test tfrecords data')
    parser.add_argument('-w', '--weights_path', type=str, help='Where you store the shadow net weights')
    return parser.parse_args()


def test_crnn(dataset_dir: str, weights_path: str, config: GlobalConfig):
    log = LogFactory.get_logger()
    is_recursive = config.get_test_config().is_recursive
    tfrecords_path = ops.join(dataset_dir, 'test_feature.tfrecords')
    if is_recursive:
        tester = RecursiveCrnnTester(tfrecords_path, weights_path, config)
    else:
        tester = BasicCrnnTester(tfrecords_path, weights_path, config)
    accuracy, distance, avg_time = tester.run()
    log.info(
        '\n* Mean test accuracy is {:.3f}\n* Mean Levenshtein edit distance is {:.3f}\n* Mean detection time for batch is {:.3f} s'
        .format(accuracy, distance, avg_time)
    )


if __name__ == '__main__':
    params = parse_params()
    config_file = params.config
    config = ConfigProvider.load_config(config_file)
    LogFactory.configure(config.get_logging_config())
    test_crnn(params.dataset_dir, params.weights_path, config)
