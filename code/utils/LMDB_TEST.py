__author__ = 'yawli'

import lmdb
import caffe
import os
import pickle
import cv2
import numpy as np

def _get_paths_from_lmdb(dataroot):
    env = lmdb.open(dataroot, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    keys_cache_file = os.path.join(dataroot, '_keys_cache.p')
    if os.path.isfile(keys_cache_file):
        print('read lmdb keys from cache: {}'.format(keys_cache_file))
        keys = pickle.load(open(keys_cache_file, "rb"))
    else:
        with env.begin(write=False) as txn:
            print('creating lmdb keys cache: {}'.format(keys_cache_file))
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, 'wb'))
    paths = sorted([key.encode('ascii') for key in keys if not key.endswith('.meta')])
    return env, paths


def get_image_paths(data_type, dataroot):
    env, paths = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            env, paths = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = []
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return env, paths


def _read_lmdb_img(env, path):
    with env.begin(write=False) as txn:
        buf = txn.get(path)#.encode('ascii'))
        #buf_meta = txn.get((path + '.meta').encode('ascii')).decode('ascii')
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(buf)
    img_flat = np.frombuffer(datum.data, dtype=np.uint8)
    #H, W, C = [int(s) for s in buf_meta.split(',')]
    #print(img_flat.shape)
    if img_flat.shape[0] == 691200:
        img = img_flat.reshape(480, 480, 3)
    else:
        img = img_flat.reshape(int(480/4), int(480/4), 3)
    return img


def read_img(env, path):
    # read image by cv2 or from lmdb
    # return: Numpy float32, HWC, BGR, [0,1]
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_lmdb_img(env, path)
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

lmdb_path = '/scratch_net/ofsoundof/yawli/Datasets/DIV2K/GT_sub_image.lmdb'  # must end with .lmdb
env, paths = get_image_paths('lmdb', lmdb_path)
from IPython import embed; embed(); exit()















