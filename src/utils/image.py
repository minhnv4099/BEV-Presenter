#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Union
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def imshow(img_or_path: Union[np.ndarray, str, Path]):
    # if isinstance(img_or_path, np.ndarray):
    #     img = Image.fromarray(img_or_path)
    # elif isinstance(img_or_path, (str, Path)):
    #     img = Image.open(img_or_path)

    plt.imshow(img_or_path)
    plt.show()
