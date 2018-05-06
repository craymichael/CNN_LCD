# =====================================================================
# dataset.py - CNNs for loop-closure detection in vSLAM systems.
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
from scipy.io import loadmat
from scipy.ndimage import imread
import numpy as np

import os
import requests
import zipfile
import sys
from glob import glob

# === DATASET VARS ===
# Data directory
DATA_DIR = 'data'
# City Centre Dataset
CITY_DATA_DIR = os.path.join(DATA_DIR, 'city')
CITY_IMGZIP_PATH = os.path.join(CITY_DATA_DIR, 'Images.zip')
CITY_IMG_PATH = os.path.join(CITY_DATA_DIR, 'Images')
CITY_GT_PATH = os.path.join(CITY_DATA_DIR, 'CityCentreGroundTruth.mat')
CITY_IMG_URL = 'http://www.robots.ox.ac.uk/~mobile/IJRR_2008_Dataset/Data/CityCentre/Images.zip'
CITY_GT_URL = 'http://www.robots.ox.ac.uk/~mobile/IJRR_2008_Dataset/Data/CityCentre/masks/CityCentreGroundTruth.mat'
# New College Dataset
COLLEGE_DATA_DIR = os.path.join(DATA_DIR, 'college')
COLLEGE_IMGZIP_PATH = os.path.join(COLLEGE_DATA_DIR, 'Images.zip')
COLLEGE_IMG_PATH = os.path.join(COLLEGE_DATA_DIR, 'Images')
COLLEGE_GT_PATH = os.path.join(COLLEGE_DATA_DIR, 'NewCollegeGroundTruth.mat')
COLLEGE_IMG_URL = 'http://www.robots.ox.ac.uk/~mobile/IJRR_2008_Dataset/Data/NewCollege/Images.zip'
COLLEGE_GT_URL = 'http://www.robots.ox.ac.uk/~mobile/IJRR_2008_Dataset/Data/NewCollege/masks/NewCollegeGroundTruth.mat'


def download_file(url, file_name):
    """Downloads a file to destination

    Code adapted from:
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url:       URL of file to download
        file_name: Where to write downloaded file
    """
    # Ensure destination exists
    dest_dir = os.path.dirname(file_name)
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    with open(file_name, 'wb') as f:
        print('Downloading {} from {}'.format(file_name, url))
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')
        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                # Output progress
                complete = dl / total_length
                done = int(50 * complete)
                sys.stdout.write('\r[{}{}] {:6.2f}%'.format('=' * done, ' ' * (50 - done),
                                                            complete * 100))
                sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()


def get_dataset(name, debug=False):
    debug_amt = 25
    if name.lower() == 'city':  # city centre dataset
        print('Loading the City Centre dataset...')
        # Load images
        print('Loading images')
        if not os.path.isfile(CITY_IMGZIP_PATH):
            download_file(CITY_IMG_URL, CITY_IMGZIP_PATH)
        if not os.path.isdir(CITY_IMG_PATH):
            # Unzip archive
            print('Unzipping {} to {}'.format(CITY_IMGZIP_PATH, CITY_DATA_DIR))
            with zipfile.ZipFile(CITY_IMGZIP_PATH, 'r') as zip_handle:
                zip_handle.extractall(CITY_DATA_DIR)
        # Sort by image number
        img_names = sorted(glob(os.path.join(CITY_IMG_PATH, '*.jpg')))
        assert len(img_names) == 2474
        if debug:
            print('Using fewer images ({}) per debug flag...'.format(
                debug_amt))
            img_names = img_names[:debug_amt]
        imgs = np.asarray([imread(img) for img in img_names])
        # Load GT
        if not os.path.isfile(CITY_GT_PATH):
            download_file(CITY_GT_URL, CITY_GT_PATH)
        print('Loading ground truth')
        gt = loadmat(CITY_GT_PATH)['truth']
        if debug:
            gt = gt[:debug_amt, :debug_amt]
    elif name.lower() == 'college':  # new college dataset
        print('Loading the New College dataset...')
        # Load images
        print('Loading images')
        if not os.path.isfile(COLLEGE_IMGZIP_PATH):
            download_file(COLLEGE_IMG_URL, COLLEGE_IMGZIP_PATH)
        if not os.path.isdir(COLLEGE_IMG_PATH):
            # Unzip archive
            print('Unzipping {} to {}'.format(COLLEGE_IMGZIP_PATH,
                                              COLLEGE_DATA_DIR))
            with zipfile.ZipFile(COLLEGE_IMGZIP_PATH, 'r') as zip_handle:
                zip_handle.extractall(COLLEGE_DATA_DIR)
        # Sort by image number
        img_names = sorted(glob(os.path.join(COLLEGE_IMG_PATH, '*.jpg')))
        assert len(img_names) == 2146
        if debug:
            print('Using fewer images ({}) per debug flag...'.format(
                debug_amt))
            img_names = img_names[:debug_amt]
        imgs = np.asarray([imread(img) for img in img_names])
        # Load GT
        if not os.path.isfile(COLLEGE_GT_PATH):
            download_file(COLLEGE_GT_URL, COLLEGE_GT_PATH)
        print('Loading ground truth')
        gt = loadmat(COLLEGE_GT_PATH)['truth']
        if debug:
            gt = gt[:debug_amt, :debug_amt]
    elif name.lower() == 'tsukuba':  # new tsukuba dataset
        raise NotImplementedError
    else:
        raise ValueError('Invalid dataset name: {}.'.format(name))
    return imgs, gt
