import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm


bone_pairs = (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),(11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    



parts = { 'train', 'test' }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bone data generation for NIA data')
    parser.add_argument('--data_path',  required=True)
    args = parser.parse_args()

    path = args.data_path
    
    for part in parts:
        print(part)
        try:
            data = np.load('../data/{}/{}_data_joint.npy'.format(path, part), mmap_mode='r')
            N, C, T, V, M = data.shape
            fp_sp = open_memmap(
                '../data/{}/{}_data_bone.npy'.format(path, part),
                dtype='float32',
                mode='w+',
                shape=(N, 2, T, V, M))

            fp_sp[:, :C, :, :, :] = data
            for v1, v2 in tqdm(bone_pairs):
                fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
        except Exception as e:
            print(f'Run into error: {e}')
            print(f'Skipping ({benchmark} {part})')