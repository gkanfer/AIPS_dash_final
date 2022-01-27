"""
Spyder Editor

This is a temporary script file.
"""

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
AIPS_object = ai.Segment_over_seed(Image_name='dmsot0273_0003-512.tif', path=path, rmv_object_nuc=0.5, block_size=59,
                                           offset=-0.0004,block_size_cyto=59, offset_cyto=-0.0004, global_ther=0.2, rmv_object_cyto=0.1,
                                           rmv_object_cyto_small=0.1, remove_border=False)

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

bf_mask = dx.binary_frame_mask(ch,nuc_s['sort_mask'])
bf_mask = np.where(bf_mask == 1, True, False)
composite[bf_mask > 0,1]=255
im_pil = Image.fromarray(composite,mode='RGB')
table_nuc = nuc_s['table']

#plt.imshow(im_pil)

for col in table_nuc.columns:
    print(col)
table_nuc['coords']
np.concatenate(np.array(table_nuc['coords']))

def composite_display(img,label_array, table,):
    fig = px.imshow(img, binary_string=True, binary_backend="png", )
    #fig.update_traces(hoverinfo="skip", hovertemplate=None)
    # for label in label_array:
    #     x = np.concatenate(np.array(table['coords']))
    #     y = np.concatenate(np.array(table['coords']))
    #     hoverinfo = (
    #             "<br>".join(
    #                 [
    #                     # All numbers are passed as floats. If there are no decimals, cast to int for visibility
    #                     f"{prop_name}: {f'{int(prop_val):d}' if prop_val.is_integer() else f'{prop_val:.3f}'}"
    #                     if np.issubdtype(type(prop_val), "float")
    #                     else f"{prop_name}: {prop_val}"
    #                     for prop_name, prop_val in
    #                     _columns].iteritems()
    #                 ]
    #             )
    #             # remove the trace name. See e.g. https://plotly.com/python/reference/#scatter-hovertemplate
    #             + " <extra></extra>"
    #     )
    # pass




from skimage import io, filters, measure, color, img_as_ubyte

img = io.imread('/Users/kanferg/Desktop/NIH_Youle/Python_projacts_general/dash/AIPS_Dash_Final/app_uploaded_files/Monocyte_no_vacuoles.jpeg', as_gray=True)[:660:2, :800:2]
label_array = measure.label(img < filters.threshold_otsu(img))
current_labels = np.unique(label_array)[np.nonzero(np.unique(label_array))]
img = img_as_ubyte(color.gray2rgb(img))
img = Image.fromarray(img)
label_array = np.pad(label_array, (1,), "constant", constant_values=(0,))
plt.imshow(label_array)