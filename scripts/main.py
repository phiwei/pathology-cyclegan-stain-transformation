import argparse
import datetime
from cyclegan.trainer import trainer

def collect_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--run_name', dest='run_name', default='train', help='name of the current run')
    parser.add_argument('--output_dir', dest='output_dir', default=None, help='models are saved here')
    parser.add_argument('--param_file_path', dest='param_file_path', default=None, help='set the parameters file path')
    parser.add_argument('--albumentations_path', dest='albumentations_path', default=None, help='augmentations file')
    parser.add_argument('--data_file_path', dest='data_file_path', default=None, help='set the data file path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = vars(collect_arguments())
    args['output_dir'] += '-{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    trainer = trainer(args)
    trainer.setup_network()
    trainer.train_network()




