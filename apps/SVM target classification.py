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


layout = html.Div(
    [
        dbc.Container(
            [
            html.H1("SVM"),
            html.Hr(),
            dcc.Tabs(id = 'tabs-svm', value = '',
                        children=[
                            dcc.Tab(label="Selection", id = "Selection-id",value="Selection-val",style={'color': 'black'},selected_style={'color': 'red'},disabled=False ),
                            dcc.Tab(label="PCA model", id = "PCA-model-id", value="PCA-model-id",style={'color': 'black'},selected_style={'color': 'red'},disabled=True),
                            dcc.Tab(label="Model generation", id = "Model-generation-id", value="Model-generation-val",style={'color': 'black'},selected_style={'color': 'red'},disabled=True),
                            dcc.Tab(label="Model test", id = "Model-test-id", value="Model-test-val",style={'color': 'black'},selected_style={'color': 'red'},disabled=True),
                    ]),
            html.Div(id='Tab_image_display'),
            dcc.Store(id='jason_ch'),
            dcc.Store(id='jason_ch2'),
            dcc.Store(id='jason_nmask2'),
            dcc.Store(id='jason_nmask4'),
            dcc.Store(id='jason_sort_mask'),
            dcc.Store(id='jason_table'),
            dcc.Store(id='jason_cell_mask_1'),
            dcc.Store(id='jason_combine'),
            dcc.Store(id='jason_cseg_mask'),
            dcc.Store(id='jason_info_table'),
            ])
    ])
# loading all the data
@app.callback([
    Output('jason_ch', 'data'),
    Output('jason_ch2', 'data'),
    Output('jason_nmask2', 'data'),
    Output('jason_nmask4', 'data'),
    Output('jason_sort_mask', 'data'),
    Output('jason_table', 'data'),
    Output('jason_cell_mask_1', 'data'),
    Output('jason_combine', 'data'),
    Output('jason_cseg_mask', 'data'),
    Output('jason_info_table', 'data')],
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
def Generate_segmentation_and_table(image,channel,high,low,bs,os,ron,bsc,osc,gt,roc,rocs):
    AIPS_object = ai.Segment_over_seed(Image_name=str(image[0]), path=DATA_PATH, rmv_object_nuc=ron,
                                       block_size=bs,
                                       offset=os,
                                       block_size_cyto=bsc, offset_cyto=osc, global_ther=gt, rmv_object_cyto=roc,
                                       rmv_object_cyto_small=rocs, remove_border=True)
    img = AIPS_object.load_image()
    if channel == 1:
        nuc_sel = '0'
        cyt_sel = '1'
    else:
        nuc_sel = '1'
        cyt_sel = '0'
    nuc_s = AIPS_object.Nucleus_segmentation(img[nuc_sel])
    ch_ = img[nuc_sel]
    # segmentation traces the nucleus segmented image based on the
    ch_ = (ch_ / ch_.max()) * 255
    ch_ = np.uint8(ch_)
    composite = np.zeros((np.shape(ch_)[0], np.shape(ch_)[1], 3), dtype=np.uint8)
    composite[:, :, 0] = ch_
    composite[:, :, 1] = ch_
    composite[:, :, 2] = ch_
    img = composite
    bf_mask = dx.binary_frame_mask(composite, nuc_s['sort_mask'])
    bf_mask = np.where(bf_mask == 1, True, False)
    composite[bf_mask > 0, 1] = 255
    label_array = nuc_s['sort_mask']
    json_object_ch = json.dumps(ch)
    json_object_ch2 = json.dumps(ch2)
    json_object_nmask2 = json.dumps(nmask2)
    json_object_nmask4 = json.dumps(nmask4)
    json_object_sort_mask = json.dumps(sort_mask)
    json_object_table = table.to_json(orient='split')
    json_object_cell_mask_1 = json.dumps(cell_mask_1)
    json_object_combine = json.dumps(combine)
    json_object_cseg_mask = json.dumps(cseg_mask)
    json_object_info_table = info_table.to_json(orient='split')
    return json_object_ch, json_object_ch2, json_object_nmask2, json_object_nmask4, json_object_sort_mask, \
           json_object_table, json_object_cell_mask_1, json_object_combine, json_object_cseg_mask, json_object_info_table


