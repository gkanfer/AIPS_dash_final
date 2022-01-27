
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

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../app_uploaded_files").resolve()
TEMP_PATH =  PATH.joinpath("../temp").resolve()

# layout to import
color_drop = dcc.Dropdown(
    id="color-drop-menu",
    options=[
        {"label": col_name.capitalize(), "value": col_name}
        for col_name in table.columns
    ],
    value="label",
)

image_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Explore object properties")),
        dbc.CardBody(
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        id="graph",
                        figure=image_with_contour(
                            img,
                            current_labels,
                            table,
                            initial_columns,
                            color_column="area",
                        ),
                    ),
                )
            )
        ),
        dbc.CardFooter(
            dbc.Row(
                [
                    dbc.Col(
                        "Use the dropdown menu to select which variable to base the colorscale on:"
                    ),
                    dbc.Col(color_drop),
                ],
                align="center",
            ),
        ),
    ]
)


layout = html.Div([
    html.Div(id='output-nuc-test')])




@app.callback(
    Output('output-nuc-test','children'),
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
    ch2_ = img[cyt_sel]
    # segmentation traces the nucleus segmented image based on the
    bf_mask = dx.binary_frame_mask(ch_, nuc_s['sort_mask'])
    bf_mask = np.where(bf_mask == 1, True, False)

    pix = af.show_image_adjust(ch_, low_prec=low, up_prec=high)
    pix = pix * 65535.000
    im_pil = Image.fromarray(np.uint16(pix))
    im_pil.save("1.png", format='png')  # this is for image processing
    encoded_image = base64.b64encode(open("1.png", 'rb').read())
    return [
            dbc.Row([
                dbc.Col([
                    dbc.Col(html.Label('Seed Image', style={'text-align': 'right'})),
                    dbc.Col(html.Img(id="img-output-orig", src='data:image/png;base64,{}'.format(encoded_image.decode()),
                                     style={'width': '100%', 'height': 'auto'}, alt="re-adjust parameters", title="ORIG"))
                             ])
                    ])]


