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

# Set up the app
external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/object_properties_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


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
im_pil = Image.fromarray(composite,mode='RGB')
plt.imshow(im_pil)

bf_mask = dx.binary_frame_mask(ch,nuc_s['sort_mask'])
bf_mask = np.where(bf_mask == 1, True, False)
composite[bf_mask > 0,:] = [50,50,50]
im_pil = Image.fromarray(composite,mode='RGB')
plt.imshow(im_pil)

bf_mask = dx.binary_frame_mask(ch,nuc_s['sort_mask'])
bf_mask = np.where(bf_mask == 1, True, False)
plt.imshow(nuc_s['sort_mask'])


mask = nuc_s['sort_mask']
bf_mask_sel = np.zeros(np.shape(mask),dtype=np.int32)
bf_mask_sel[mask==147]=147
plt.imshow(bf_mask_sel)
bf_mask = dx.binary_frame_mask(ch,bf_mask_sel)
bf_mask = np.where(bf_mask == 1, True, False)
plt.imshow(bf_mask)

from scipy.ndimage.morphology import binary_opening, binary_erosion, binary_dilation
info_table = pd.DataFrame(
    measure.regionprops_table(
        bf_mask_sel,
        intensity_image=ch,
        properties=['area', 'label', 'centroid'],
    )).set_index('label')
info_table['label'] = range(2, len(info_table) + 2)
if len(info_table) < 2:
    seg_mask_eros_9 = binary_erosion(bf_mask_sel, structure=np.ones((9, 9))).astype(np.float64)
    seg_mask_eros_3 = binary_erosion(bf_mask_sel, structure=np.ones((3, 3))).astype(np.float64)
    seg_frame = np.where(seg_mask_eros_9 + seg_mask_eros_3 == 2, 3, seg_mask_eros_3)
    framed_mask = np.where(seg_frame == 3, 0, seg_mask_eros_3)
    plt.imshow(framed_mask)


for i in list(info_table.index.values):
    seg_mask_temp = np.where(bf_mask_sel == 3, 0,  nuc_s['sort_mask'],)
    plt.imshow(seg_mask_temp)
    seg_mask_eros_9 = binary_erosion(seg_mask_temp, structure=np.ones((9, 9))).astype(np.float64)
    seg_mask_eros_3 = binary_erosion(seg_mask_temp, structure=np.ones((3, 3))).astype(np.float64)
    seg_frame = np.where(seg_mask_eros_9 + seg_mask_eros_3 == 2, 3, seg_mask_eros_3)
    framed_mask = np.where(seg_frame == 3, 0, seg_mask_eros_3)

cseg_mask = seg['cseg_mask']
bf_mask_sel = np.zeros(np.shape(cseg_mask),dtype=np.int32)
bf_mask_sel[cseg_mask == 147] = 147
plt.imshow(bf_mask_sel)
c_mask = dx.binary_frame_mask_single_point(bf_mask_sel)
c_mask = np.where(c_mask == 1, True, False)
plt.imshow(c_mask)
rgb_input_img = np.zeros((np.shape(ch2)[0], np.shape(ch2)[1], 3), dtype=np.uint8)
rgb_input_img[:, :, 0] = ch2
rgb_input_img[:, :, 1] = ch2
rgb_input_img[c_mask > 0, 2] = 255
img = img_as_ubyte(color.gray2rgb(rgb_input_img))
img_input_rgb_pil = Image.fromarray(img)
plt.imshow(img_input_rgb_pil)