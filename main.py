from __future__ import print_function
import os
import argparse
from glob import glob
import numpy as np
import tensorflow as tf

from model import lowlight_enhance
from utils import *

# Ensure TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser(description='Low-Light Image Enhancement')

parser.add_argument('--use_gpu', type=int, default=1, help='1 for GPU, 0 for CPU')
parser.add_argument('--gpu_idx', default="0", help='GPU index')
parser.add_argument('--gpu_mem', type=float, default=0.5, help="GPU memory usage (0 to 1)")
parser.add_argument('--phase', default='train', help='train or test')

parser.add_argument('--epoch', type=int, default=100, help='Number of total epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--patch_size', type=int, default=48, help='Patch size')
parser.add_argument('--start_lr', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--eval_every_epoch', type=int, default=20, help='Checkpoint save frequency')

parser.add_argument('--checkpoint_dir', default='./checkpoint', help='Checkpoint directory')
parser.add_argument('--sample_dir', default='./sample', help='Sample output directory')

parser.add_argument('--save_dir', default='./test_results', help='Test output directory')
parser.add_argument('--test_dir', default='./data/test/low', help='Test input directory')
parser.add_argument('--decom', type=int, default=0, help='0 for enhancement only, 1 for decomposition')

args = parser.parse_args()


def lowlight_train(lowlight_enhance):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # Learning rate schedule
    lr = args.start_lr * np.ones([args.epoch])
    lr[20:] = lr[0] / 10.0

    # Load training images
   # train_low_data_names = glob('./data/our485/low/*.png') + glob('./data/syn/low/*.png')
    #train_high_data_names = glob('./data/our485/high/*.png') + glob('./data/syn/high/*.png')

        #train_low_data_names = glob('./data/our485/low/*.png') + glob('./data/syn/low/*.png')
    train_low_data_names = glob ('content/drive/My Drive/Enhan/data/our485/low/*.png') + glob('content/drive/My Drive/Enhan/data/syn/low/*.png')

   
    train_low_data_names.sort()
    #train_high_data_names = glob('./data/our485/high/*.png') + glob('./data/syn/high/*.png')
    train_high_data_names = glob('content/drive/My Drive/Enhan/data/our485/high/*.png') + glob('content/drive/My Drive/Enhan/data/syn/high/*.png')

    train_low_data_names.sort()
    train_high_data_names.sort()

    assert len(train_low_data_names) == len(train_high_data_names)
    print(f'[*] Number of training images: {len(train_low_data_names)}')

    train_low_data = [load_images(img) for img in train_low_data_names]
    train_high_data = [load_images(img) for img in train_high_data_names]

    eval_low_data = [load_images(img) for img in glob('./content/drive/My Drive/Enhan/data/eval/low/*.*')]

    # Train the model
    lowlight_enhance.train(
        train_low_data, train_high_data, eval_low_data,
        batch_size=args.batch_size, patch_size=args.patch_size,
        epoch=args.epoch, lr=lr, sample_dir=args.sample_dir,
        ckpt_dir=os.path.join(args.ckpt_dir, 'Decom'),
        eval_every_epoch=args.eval_every_epoch, train_phase="Decom"
    )

    lowlight_enhance.train(
        train_low_data, train_high_data, eval_low_data,
        batch_size=args.batch_size, patch_size=args.patch_size,
        epoch=args.epoch, lr=lr, sample_dir=args.sample_dir,
        ckpt_dir=os.path.join(args.ckpt_dir, 'Relight'),
        eval_every_epoch=args.eval_every_epoch, train_phase="Relight"
    )


def lowlight_test(lowlight_enhance):
    if args.test_dir is None:
        print("[!] Please provide --test_dir")
        exit(0)

    os.makedirs(args.save_dir, exist_ok=True)

    test_low_data_names = glob(os.path.join(args.test_dir, '*.*'))
    test_low_data = [load_images(img) for img in test_low_data_names]

    lowlight_enhance.test(test_low_data, [], test_low_data_names, save_dir=args.save_dir, decom_flag=args.decom)


def main(_):
    if args.use_gpu:
        print("[*] Using GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] Using CPU")
        with tf.compat.v1.Session() as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)


if __name__ == '__main__':
    main(None)  # Replace `tf.app.run()` with `main(None)`
