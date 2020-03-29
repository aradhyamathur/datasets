
"""TODO(pix3d): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets.public_api as tfds
import os
import tensorflow as tf
import numpy as np
from skimage import io
from skimage import color

# TODO(pix3d): BibTeX citation
_CITATION = """
@inproceedings{sun2018pix3d,
  title={Pix3d: Dataset and methods for single-image 3d shape modeling},
  author={Sun, Xingyuan and Wu, Jiajun and Zhang, Xiuming and Zhang, Zhoutong and Zhang, Chengkai and Xue, Tianfan and Tenenbaum, Joshua B and Freeman, William T},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2974--2983},
  year={2018}
}
"""

# TODO(pix3d):
_DESCRIPTION = """
The authors study 3D shape modeling from a single image and makecontributions to\
 it in several aspects. They present Pix3D, a large-scale benchmark of diverse\
 image-shape pairs withpixel-level 2D-3D alignment. Pix3D has wide applications\
 inshape-related tasks including reconstruction, retrieval, view-point\
 estimation,etc. 
"""


class Pix3d(tfds.core.GeneratorBasedBuilder):
  """TODO(pix3d): Short description of my dataset."""

  # TODO(pix3d): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # TODO(pix3d): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            "image" : tfds.features.Image(),
            # "model" : tfds.features.ModelObj(),
            "mask" : tfds.features.Image(),
            # "label" : tfds.features.ClassLabel(num_classes=9)
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=("image", "mask"),
        # Homepage of the dataset for documentation
        homepage='http://pix3d.csail.mit.edu/',
        citation=_CITATION,
    )
  
  def sort_paths(self, paths):
    paths = [(int(p.split('/')[-1].split('.')[0]), p) for p in paths]
    paths = sorted(paths, key=lambda x:x[0])
    paths = [p[1] for p in paths]
    return paths

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # TODO(pix3d): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    extracted_path = dl_manager.download_and_extract({
      'pix3d' : 'http://pix3d.csail.mit.edu/data/pix3d.zip'
      })
    extracted_path = extracted_path['pix3d']
    classes = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'tool', 'table', 'wardrobe']
    img_paths = []
    for v in classes:
      sub_paths = []
      for path in os.listdir(os.path.join(extracted_path, 'img', v)):
        sub_paths.append(os.path.join(extracted_path, 'img', v, path))
      sub_paths = self.sort_paths(sub_paths)
      img_paths.extend(sub_paths)
    
    mask_paths = []
    for v in classes:
      sub_paths = []
      for path in os.listdir(os.path.join(extracted_path, 'mask', v)):
        sub_paths.append(os.path.join(extracted_path, 'mask', v, path))
      sub_paths = self.sort_paths(sub_paths)
      mask_paths.extend(sub_paths)

    assert len(img_paths) == len(mask_paths)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
              'paths': list(zip(img_paths, mask_paths))

            # 'model':  TODO
            },
        ),
    ]

  def _generate_examples(self, paths):
    """Yields examples."""

    for index, (path_img, path_mask) in enumerate(paths):
      img_ext = path_img.split('/')[-1].split('.')[-1]
      mask_ext = path_mask.split('/')[-1].split('.')[-1]
      
      if (mask_ext != 'tiff') and ('tiff' != img_ext):
        mask = np.array(tf.keras.preprocessing.image.load_img(path_mask, color_mode='rgb'))
        img = np.array(tf.keras.preprocessing.image.load_img(path_img, color_mode='rgb'))
        
        yield index, {"image": img, 'mask': mask}

