# =====================================================================
# cnn_models.py - CNNs for loop-closure detection in vSLAM systems.
# Copyright (C) 2018  Zach Carmichael
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =====================================================================
from skimage.transform import resize as skresize
import numpy as np

import sys
import os
import tarfile

# local imports
from dataset import download_file

if sys.version_info.major == 2:  # .major requires 2.7+
    print('Python 2 detected: OverFeat-only mode.')
    # OverFeat Architecture
    import overfeat
elif sys.version_info.major == 3:
    print('Python 3 detected: OverFeat unavailable.')
    # Add local library to system path
    sys.path.append(os.path.join('models', 'research', 'slim'))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # TF Slim Models
    from models.research.slim.preprocessing.preprocessing_factory import get_preprocessing
    from models.research.slim.nets.nets_factory import get_network_fn
    import tensorflow as tf  # For TF models

    slim = tf.contrib.slim
else:
    raise Exception('how: {}'.format(sys.version_info.major))

# === MODEL VARS ===
# Model checkpoint directory
CKPT_DIR = 'tf_ckpts'
# Inception V1
INCEPTION_V1_URL = 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz'
INCEPTION_V1_PATH = os.path.join(CKPT_DIR, 'inception_v1_2016_08_28.tar.gz')
INCEPTION_V1_CKPT = os.path.join(CKPT_DIR, 'inception_v1.ckpt')
# Inception V2
INCEPTION_V2_URL = 'http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz'
INCEPTION_V2_PATH = os.path.join(CKPT_DIR, 'inception_v2_2016_08_28.tar.gz')
INCEPTION_V2_CKPT = os.path.join(CKPT_DIR, 'inception_v2.ckpt')
# Inception V3
INCEPTION_V3_URL = 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz'
INCEPTION_V3_PATH = os.path.join(CKPT_DIR, 'inception_v3_2016_08_28.tar.gz')
INCEPTION_V3_CKPT = os.path.join(CKPT_DIR, 'inception_v3.ckpt')
# Inception V4
INCEPTION_V4_URL = 'http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz'
INCEPTION_V4_PATH = os.path.join(CKPT_DIR, 'inception_v4_2016_09_09.tar.gz')
INCEPTION_V4_CKPT = os.path.join(CKPT_DIR, 'inception_v4.ckpt')
# NASNet
NASNET_URL = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz'
NASNET_PATH = os.path.join(CKPT_DIR, 'nasnet-a_large_04_10_2017.tar.gz')
NASNET_CKPT = os.path.join(CKPT_DIR, 'model.ckpt')
NASNET_CKPT_FULL = os.path.join(CKPT_DIR, 'model.ckpt.data-00000-of-00001')
# ResNet V2 152
RESNET_V2_152_URL = 'http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz'
RESNET_V2_152_PATH = os.path.join(CKPT_DIR, 'resnet_v2_152_2017_04_14.tar.gz')
RESNET_V2_152_CKPT = os.path.join(CKPT_DIR, 'resnet_v2_152.ckpt')
# === MODEL INFO ===
DEFAULT_FEATURE_LAYER = {
    'inception_v1': 'InceptionV1/Logits/AvgPool_0a_7x7/AvgPool:0',
    'inception_v2': 'InceptionV2/Logits/AvgPool_1a_7x7/AvgPool:0',
    'inception_v3': 'InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0',
    'inception_v4': 'InceptionV4/Logits/AvgPool_1a/AvgPool:0',
    'nasnet_large': 'final_layer/Mean:0',
    'resnet_v2_152': 'resnet_v2_152/pool5:0',
    'overfeat_0': 19,
    'overfeat_1': 20
}
MODEL_PARAMS_NAME = {
    'inception_v1': 'InceptionV1',
    'inception_v2': 'InceptionV2',
    'inception_v3': 'InceptionV3',
    'inception_v4': 'InceptionV4',
    'nasnet_large': None,
    'resnet_v2_152': 'resnet_v2_152',
    'overfeat_0': None,
    'overfeat_1': None
}
MODEL_CKPT_PATHS = {  # Slim-only
    'inception_v1': [INCEPTION_V1_CKPT, INCEPTION_V1_URL, INCEPTION_V1_PATH],
    'inception_v2': [INCEPTION_V2_CKPT, INCEPTION_V2_URL, INCEPTION_V2_PATH],
    'inception_v3': [INCEPTION_V3_CKPT, INCEPTION_V3_URL, INCEPTION_V3_PATH],
    'inception_v4': [INCEPTION_V4_CKPT, INCEPTION_V4_URL, INCEPTION_V4_PATH],
    'nasnet_large': [(NASNET_CKPT_FULL, NASNET_CKPT), NASNET_URL, NASNET_PATH],
    'resnet_v2_152': [RESNET_V2_152_CKPT, RESNET_V2_152_URL, RESNET_V2_152_PATH]
}
MODEL_ALIASES = {
    'nasnet': 'nasnet_large',
    'overfeat': 'overfeat_1'
}


def get_ckpt(url, dl_dest):
    """Downloads and extracts model checkpoint file."""
    download_file(url, dl_dest)
    with tarfile.open(dl_dest) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=CKPT_DIR)


def _capitalize(s):
    s = s.split('_')
    for i, ss in enumerate(s):
        if len(ss):
            ss = ss[0].upper() + ss[1:]
        s[i] = ss
    return ' '.join(s)


def _resolve_alias(name):
    name = name.lower()
    return MODEL_ALIASES.get(name, name)


def is_valid_model(model_name):
    if sys.version_info.major == 2 and model_name != 'overfeat':
        print('Python 3 needed for TF Slim model (execute using python3 not python2...)')
        sys.exit(1)

    if model_name[:len('overfeat')] == 'overfeat':
        if sys.version_info.major == 3:
            print('Python 2 needed for overfeat model (execute using python2 not python3...)')
            sys.exit(1)

    return _resolve_alias(model_name) in DEFAULT_FEATURE_LAYER


def valid_model_names():
    return DEFAULT_FEATURE_LAYER.keys()


def is_tf_model(model_name):
    return _resolve_alias(model_name) in MODEL_CKPT_PATHS


def overfeat_preprocess(img, resize):
    # Ensure single-precision
    img = img
    # Crop and resize image
    h0, w0 = img.shape[:2]
    # Compute crop indices
    d0 = min(h0, w0)
    hc = round((h0 - d0) / 2.)
    wc = round((w0 - d0) / 2.)
    # Center crop image (ensure 3 channels...)
    img = img[int(hc):int(hc + d0), int(wc):int(wc + d0), :]
    # Resize image
    img = skresize(img, (resize, resize), mode='constant',
                   preserve_range=True, order=1).astype(np.float32)
    # Change channel order: h,w,c -> c,h,w
    img = np.rollaxis(img, 2, 0)
    return img


def get_overfeat_features(imgs, weights_path, typ, layer=None, cache=None):
    """Returns features at layer for given image(s) from OverFeat model.

    Small (fast) network: 22 layers
    Large (accurate) network: 25 layers

    Args:
        imgs:         Iterable of images each of shape (h,w,c)
        weights_path: Path to the OverFeat weights
        typ:          0 for small, 1 for large version of OverFeat
        layer:        The layer to extract features from
        cache:        Dict containing descs/other cached values
    """
    if cache is None:
        cache = {}
    if 'overfeat_descs' not in cache:
        # Initialize network
        print('Loading OverFeat ({}) model...'.format(typ))
        overfeat.init(weights_path, typ)
        # Determine feature layer if none specified
        if layer is None:
            if overfeat.get_n_layers() == 22:  # small
                layer = 19  # 16 also recommended
            else:  # large
                # Layer used by Zhang et al.
                layer = 22
        # Determine resize dim
        if typ == 0:
            resize = 231  # small network
        else:
            resize = 221  # large network
        # Allocate for feature descriptors
        descs = []
        # Run images through network
        print('Running images through OverFeat, extracting features '
              'at layer {}.'.format(layer))

        for idx, img in enumerate(imgs):
            if (idx + 1) % 100 == 0:
                print('Processing image {}...'.format(idx + 1))
            # Preprocess image
            img = overfeat_preprocess(img, resize)
            # Run through model
            _ = overfeat.fprop(img)
            # Retrieve feature output
            desc = overfeat.get_output(layer)
            descs.append(desc)
        # Free network
        overfeat.free()
        # NumPy-ify
        descs = np.asarray(descs)
        cache.update(overfeat_descs=descs)
    else:
        descs = cache['overfeat_descs']
    return descs, cache


def get_slim_model_features(imgs, model_name, layer=None, cache=None):
    """Returns features at layer for given image(s) from a TF-Slim model.

    Args:
        imgs:       Iterable of images each of shape (h,w,c)
        model_name: The model name
        layer:      The layer to extract features from
        cache:      Dict containing descs/other cached values
    """
    model_name = _resolve_alias(model_name)

    if cache is None:
        cache = {}
    if model_name + '_descs' not in cache:
        # Get model ckpt info
        model_ckpt, model_url, model_path = MODEL_CKPT_PATHS[model_name]

        if isinstance(model_ckpt, tuple):
            # For ckpts that should be specified using file basename only (no extension)
            model_ckpt, model_ckpt_full = model_ckpt
        else:
            model_ckpt = model_ckpt_full = model_ckpt

        # Grab ckpt if not available locally
        if not os.path.isfile(model_ckpt_full):
            get_ckpt(model_url, model_path)

        # Determine feature layer if none specified
        if layer is None:
            layer = DEFAULT_FEATURE_LAYER[model_name]

        # Allocate for feature descriptors
        descs = []
        # Run images through network
        print('Running images through {}, extracting features '
              'at layer {}.'.format(_capitalize(model_name), layer))

        # Set up image placeholder
        net_fn = get_network_fn(model_name, num_classes=None)
        im_size = net_fn.default_image_size
        image = tf.placeholder(tf.float32, shape=(None, None, 3))

        # Preprocess image
        preprocess_func = get_preprocessing(model_name, is_training=False)
        pp_image = preprocess_func(image, im_size, im_size)
        pp_images = tf.expand_dims(pp_image, 0)

        # Compute network output
        logits_input, _ = net_fn(pp_images)  # because num_classes=None

        # Restore parameters from checkpoint as init_fn
        model_params_name = MODEL_PARAMS_NAME[model_name]
        if model_params_name:
            model_vars = slim.get_model_variables(model_params_name)
        else:
            model_vars = slim.get_model_variables()
        init_fn = slim.assign_from_checkpoint_fn(
            model_ckpt, model_vars)

        with tf.Session() as sess:
            # Init model variables
            init_fn(sess)
            # Get target feature tensor
            feat_tensor = sess.graph.get_tensor_by_name(layer)

            for idx, img in enumerate(imgs):
                if (idx + 1) % 100 == 0:
                    print('Processing image {}...'.format(idx + 1))
                # Run image through model
                desc = sess.run(feat_tensor,
                                feed_dict={image: img})
                descs.append(desc)

        descs = np.asarray(descs)
        # Update cache
        cache.update({model_name + '_descs': descs})
    else:
        descs = cache[model_name + '_descs']
    return descs, cache


def get_model_features(imgs, model_name, overfeat_weights_path=None,
                       overfeat_typ=None, layer=None, cache=None):
    """ Get model features from an available model.

    Args:
        imgs:                  The images to extract features for
        model_name:            Name of the CNN (see is_valid_model)
        overfeat_weights_path: See get_overfeat_features
        overfeat_typ:          See get_overfeat_features
        layer:                 See get_overfeat_features or get_slim_model_features
        cache:                 See get_overfeat_features or get_slim_model_features

    Returns:
        descs, cache: See get_overfeat_features or get_slim_model_features
    """
    if is_valid_model(model_name):
        if is_tf_model(model_name):
            return get_slim_model_features(imgs, model_name, layer=layer, cache=cache)
        else:
            return get_overfeat_features(imgs, overfeat_weights_path, overfeat_typ,
                                         layer=layer, cache=cache)
    else:
        raise ValueError('`{}` is not a valid model name. Valid:\n{}.'.format(model_name,
                                                                              valid_model_names()))
