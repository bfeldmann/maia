import os
import sys
import json
import cv2
import numpy as np
import torch
from torchvision import transforms as pth_transforms
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from PIL import Image
from PIL import UnidentifiedImageError
# Currently unused, might need to change parts that use PIL to VipsImage instead
#from pyvips import Image as VipsImage
#from pyvips.error import Error as VipsError

class bc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    TEST = '\033[97m'

# Simplified preprocessing
preprocess_simple = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

# Generates feature maps using DINO and saves them as .npz.
class DinoFVGenerator(object):
    def __init__(self, params):
        # Architectures: See https://github.com/facebookresearch/dino#pretrained-models-on-pytorch-hub
        # E.g.: dino_resnet50, dino_vitb8, dino_vitb16, dino_vits8, dino_vits16
        self.backbone = params['backbone']

        # Dict of cropped image IDs and file paths to the cropped images to process.
        self.cropped_images = params['cropped_images']

        # Path to cropped images, ensure path ends in separator
        self.cropped_images_path = params['cropped_images_path'] if params['cropped_images_path'].endswith(os.sep) else params['cropped_images_path']+os.sep

        # Path to feature vector save directory
        self.fv_path = '{}/fv'.format(self.tmp_dir)

        self.max_workers = params['max_workers']

        # Dimension used for resizing cropped images.
        self.resize_dimension = params['resize_dimension']

        # Path to the directory to store temporary files.
        self.tmp_dir = params['tmp_dir']

    def generate(self):
        # Check if dir(s) are there, create if missing
        self.ensure_dirs()

        # TODO: Use already existing model, if available, download otherwise. Currently downloading every time
        model = torch.hub.load('facebookresearch/dino:main', self.backbone)

        suffix = '.simple_{}_{}x{}'.format(backbone, self.resize_dimension, self.resize_dimension)

        #executor = ThreadPoolExecutor(max_workers=self.max_workers)
        executor = ProcessPoolExecutor(max_workers=self.max_workers)
        l = len(images)
        jobs = []
        for i, image in enumerate(self.cropped_images):
            path = self.cropped_images_path+image
            print(f'{i+1}/{l} Processing image: {path}')
            try:
                features = getFeatures(model, path)
                jobs.append(executor.submit(saveFeatures, self.fv_path, path, features, suffix))
            except Exception as e:
                print(f'{bc.FAIL}Exception during processing of image {i+1}/{l}: {e}{bc.ENDC}')
        executor.shutdown(wait=True)

        fv_files = []
        for job in as_completed(jobs):
            fv = job.result()
            if fv is not False:
                fv_files.extend(i)

        if len(fv_files) == 0:
            raise Exception('No feature vectors in dataset. All corrupt?')

        return {
            'fv_path': self.fv_path,
            'fv_files': fv_files,
            'resize_dimension': self.resize_dimension,
        }

    # Ensure path exists
    def ensure_dirs(self):
        if not os.path.exists(self.fv_path):
           os.makedirs(self.fv_path)

    # Gets the features from the given DINO model
    def getFeatures(model, path):
        features = None
        try:
            img = Image.open(path)
            features = (model(preprocess_simple(img).unsqueeze(0))).detach().numpy()[0]
            img.close()
        except FileNotFoundError:
            print(bc.FAIL+'Cannot find file: '+path+bc.ENDC)
            return False
        except UnidentifiedImageError:
            print(bc.FAIL+'Cannot open image: '+path+bc.ENDC)
            return False
        return np.reshape(features, (1, len(features)))

    # Saves the features as .npz.
    def saveFeatures(target, path, features, suffix='.dino'):
        np.savez_compressed(target+os.path.basename(path)+suffix, features=features)

with open(sys.argv[1]) as f:
    params = json.load(f)

runner = DinoFVGenerator(params)
output = runner.generate()

with open(params['output_path'], 'w') as f:
    json.dump(output, f)
