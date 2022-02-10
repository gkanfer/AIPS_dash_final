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
from scipy.ndimage.morphology import binary_opening, binary_erosion, binary_dilation
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
from utils.Display_composit import image_with_contour, countor_map

path = '/Users/kanferg/Desktop/NIH_Youle/Python_projacts_general/dash/AIPS_Dash_Final/app_uploaded_files/'
#Composite.tif10.tif
AIPS_object = ai.Segment_over_seed(Image_name='dmsot0273_0003-512.tif', path=path, rmv_object_nuc=0.5, block_size=83,
                                           offset=0.00001,block_size_cyto=11, offset_cyto=-0.0003, global_ther=0.4, rmv_object_cyto=0.99,
                                           rmv_object_cyto_small=0.9, remove_border=False)


AIPS_object = ai.Segment_over_seed(Image_name='dmsot0273_0003-512.tif', path=path, rmv_object_nuc=0.99, block_size=83,
                                           offset=0.01,block_size_cyto=13, offset_cyto=-0.0003, global_ther=0.4, rmv_object_cyto=0.99,
                                           rmv_object_cyto_small=0.9, remove_border=False)

img = AIPS_object.load_image()
nuc_s = AIPS_object.Nucleus_segmentation(img['1'], inv=False)
ch = img['1']
ch2 = img['0']
seg = AIPS_object.Cytosol_segmentation( ch,ch2,nuc_s['sort_mask'],nuc_s['sort_mask_bin'])
cseg_mask = seg['cseg_mask']
np.unique(cseg_mask)
plt.imshow(cseg_mask)

sort_mask = nuc_s['sort_mask']
plt.imshow(sort_mask)
len(np.unique(sort_mask))



test = af.rgb_file_gray_scale(ch2,mask=cseg_mask,channel=1)
plt.imshow(test)

ch2 = (ch2 / ch2.max()) * 255
ch2_u8 = np.uint8(ch2)
rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
rgb_input_img[:, :, 0] = ch2_u8
rgb_input_img[:, :, 1] = ch2_u8
rgb_input_img[:, :, 2] = ch2_u8
bf_mask = dx.binary_frame_mask(ch2_u8, seg['sort_mask_sync'])
bf_mask = np.where(bf_mask == 1, True, False)
rgb_input_img[bf_mask > 0, 2] = 255
valuectrl = [87,133,145]
valuectrl = []
#valuetarget = [164,203,223]
valuetarget = [203,223]
cseg_mask = seg['cseg_mask']


rgb = countor_map(cseg_mask,valuectrl,valuetarget,rgb_input_img)
plt.imshow(rgb)

bf_mask_sel_ctrl = np.zeros(np.shape(cseg_mask), dtype=np.int32)
for list in valuectrl:
    bf_mask_sel_ctrl[cseg_mask == list] = list
plt.imshow(bf_mask_sel_ctrl)


bf_mask_sel_trgt = np.zeros(np.shape(cseg_mask), dtype=np.int32)
for list_ in valuetarget:
    bf_mask_sel_trgt[cseg_mask == list_] = list_
np.unique(bf_mask_sel_trgt)

0.01*100
