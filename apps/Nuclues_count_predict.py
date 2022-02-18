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
from utils.Display_composit import image_with_contour
from utils import AIPS_functions as af
from utils import AIPS_module as ai
from utils import display_and_xml as dx

import pathlib
from app import app

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../app_uploaded_files").resolve()
TEMP_PATH = PATH.joinpath("../temp").resolve()

# layout to import

layout = html.Div(
    [
        dbc.Container(
            [dbc.Row([
                dbc.Col(children=[
                    html.Div(id='output-nuc-image')
                ])
                ]),
            ]
        )
        ]
    )
@app.callback(
    Output('output-nuc-image','children'),
    [Input('upload-image', 'filename'),
    State('act_ch', 'value'),
    State('high_pass', 'value'),
    State('low_pass', 'value'),
    State('block_size','value'),
    State('offset','value'),
    State('rmv_object_nuc','value'),
    State('block_size_cyto', 'value'),
    State('offset_cyto', 'value'),
    State('global_ther', 'value'),
    State('rmv_object_cyto', 'value'),
    State('rmv_object_cyto_small', 'value')
     ])
def update_nuc(image,channel,high,low,bs,os,ron,bsc,osc,gt,roc,rocs):
    AIPS_object = ai.Segment_over_seed(Image_name=str(image[0]), path=DATA_PATH, rmv_object_nuc=ron,
                                       block_size=bs,
                                       offset=os,
                                       block_size_cyto=bsc, offset_cyto=osc, global_ther=gt, rmv_object_cyto=roc,
                                       rmv_object_cyto_small=rocs, remove_border=False)
    img = AIPS_object.load_image()
    if channel == 1:
        nuc_sel = '0'
        cyt_sel = '1'
    else:
        nuc_sel = '1'
        cyt_sel = '0'
    nuc_s = AIPS_object.Nucleus_segmentation(img[nuc_sel])
    sort_mask = nuc_s['sort_mask']
    ch_ = img[nuc_sel]
    # segmentation traces the nucleus segmented image based on the
    ch1_sort_mask = af.rgb_file_gray_scale(ch_, mask=sort_mask, channel=2)
    fig_im_pil_sort_mask = af.px_pil_figure(ch1_sort_mask, bit=3, mask_name='sort_mask', fig_title='RGB map - seed', wh=700)
    return [
        dbc.Row([
            dbc.Col(
                dcc.Graph(
                    id="Nuclues_pick",
                    figure=fig_im_pil_sort_mask))
            ])]


