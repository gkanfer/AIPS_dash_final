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
    current_labels = np.unique(label_array)[np.nonzero(np.unique(label_array))]
    prop_names = [
        "label",
        "area",
        "perimeter",
        "eccentricity",
        "euler_number",
        "mean_intensity",
    ]
    prop_table = measure.regionprops_table(
        label_array, intensity_image=composite, properties=prop_names)
    table = pd.DataFrame(prop_table)
    # Format the Table columns
    columns = [
        {"name": label_name, "id": label_name, "selectable": True}
        if precision is None
        else {
            "name": label_name,
            "id": label_name,
            "type": "numeric",
            "selectable": True,
        }
        for label_name, precision in zip(prop_names, (None, None, 4, 4, None, 3))]
    initial_columns = ["label", "area"]
    #img = img_as_ubyte(color.gray2rgb(composite))
    img = Image.fromarray(img)
    label_array = np.pad(label_array, (1,), "constant", constant_values=(0,))
    return [
         dbc.CardHeader(html.H2("Explore seed properties", style={'text-align': 'center'})),
            dbc.CardBody(
                dbc.Row(
                    dbc.Col(
                        dcc.Graph(
                            id="graph",
                            figure=image_with_contour(
                                img,
                                label_array,
                                table,
                                initial_columns,
                                color_column="area",
                            ),
                        ),
                    )
                )
            )]



