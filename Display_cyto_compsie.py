import dash_daq as daq
import json
import dash
import dash.exceptions
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State
import tifffile as tfi
import glob
import os
import numpy as np
from skimage.exposure import rescale_intensity, histogram
import matplotlib.pyplot as plt
import matplotlib as mpl

from PIL import Image, ImageEnhance
import base64
import pandas as pd
import re
from random import randint
from io import BytesIO
from flask_caching import Cache
from dash.long_callback import DiskcacheLongCallbackManager
import plotly.express as px
from skimage import io, filters, measure, color, img_as_ubyte

from utils.controls import controls, controls_nuc, controls_cyto
from utils import AIPS_functions as af
from utils import AIPS_module as ai
from utils import display_and_xml as dx

import pathlib
from app import app
from utils.controls import controls, controls_nuc, controls_cyto
from utils import AIPS_functions as af
from utils import AIPS_module as ai
from utils import display_and_xml as dx



path = '/Users/kanferg/Desktop/NIH_Youle/Python_projacts_general/dash/AIPS_Dash_Final/app_uploaded_files/'
#Composite.tif10.tif
AIPS_object = ai.Segment_over_seed(Image_name='dmsot0273_0003-512.tif', path=path, rmv_object_nuc=0.5, block_size=83,
                                           offset=0.00001,block_size_cyto=11, offset_cyto=-0.0003, global_ther=0.4, rmv_object_cyto=0.99,
                                           rmv_object_cyto_small=0.2, remove_border=False)

img = AIPS_object.load_image()
nuc_s = AIPS_object.Nucleus_segmentation(img['1'], inv=False)
ch = img['1']
ch2 = img['0']
#ch2 = ch2*2**16
ch2 = (ch2/ch2.max())*255
ch2 = np.uint8(ch2)
composite = np.zeros((np.shape(ch2)[0],np.shape(ch2)[1],3),dtype=np.uint8)
composite[:,:,0] = ch2
composite[:,:,1] = ch2
composite[:,:,2] = ch2
plt.imshow(nuc_s['sort_mask'])
bf_mask = dx.binary_frame_mask(ch,nuc_s['sort_mask'])
bf_mask = np.where(bf_mask == 1, True, False)
composite[bf_mask > 0,1]=255
im_pil = Image.fromarray(composite,mode='RGB')
table_nuc = nuc_s['table']

#plt.imshow(im_pil)
#segmant cytosol


seg = AIPS_object.Cytosol_segmentation( ch,ch2,nuc_s['sort_mask'],nuc_s['sort_mask_bin'])
plt.imshow(seg['cseg_mask'])
bf_mask = dx.binary_frame_mask(ch,seg['sort_mask_sync'])
bf_mask = np.where(bf_mask == 1, True, False)
c_mask = dx.binary_frame_mask(ch,seg['cseg_mask'])
c_mask = np.where(c_mask == 1, True, False)

composite[bf_mask > 0,1]=255
composite[c_mask > 0,0]=255
im_pil = Image.fromarray(composite,mode='RGB')
plt.imshow(im_pil)


ch2 = (ch2/ch2.max())*255
ch2 = np.uint8(ch2)
composite = np.zeros((np.shape(ch2)[0],np.shape(ch2)[1],3),dtype=np.uint8)
composite[:,:,0] = ch2
composite[:,:,1] = ch2
composite[:,:,2] = ch2

bf_mask = dx.binary_frame_mask(ch,nuc_s['sort_mask'])
bf_mask = np.where(bf_mask == 1, True, False)
composite[bf_mask > 0,1]=255
im_pil = Image.fromarray(composite,mode='RGB')
table_nuc = nuc_s['table']

prop_names = [
    "label",
    "area",
    "eccentricity",
    "euler_number",
    "extent",
    "feret_diameter_max",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "moments",
    "moments_central",
    "moments_hu",
    "moments_normalized",
    "orientation",
    "perimeter",
    "perimeter_crofton",
    "slice",
    "solidity"
]
table_prop = measure.regionprops_table(
    seg['cseg_mask'], intensity_image=composite, properties=prop_names
)