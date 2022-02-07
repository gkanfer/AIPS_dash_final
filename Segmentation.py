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



# Set up the app
external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/object_properties_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# make image and Mask
path = '/Users/kanferg/Desktop/NIH_Youle/Python_projacts_general/dash/AIPS_Dash_Final/app_uploaded_files/'
#Composite.tif10.tif
AIPS_object = ai.Segment_over_seed(Image_name='dmsot0273_0003-512.tif', path=path, rmv_object_nuc=0.5, block_size=83,
                                           offset=0.00001,block_size_cyto=11, offset_cyto=-0.0003, global_ther=0.4, rmv_object_cyto=0.99,
                                           rmv_object_cyto_small=0.2, remove_border=False)

img = AIPS_object.load_image()
nuc_s = AIPS_object.Nucleus_segmentation(img['1'], inv=False)
ch = img['1']
ch2 = img['0']
seg = AIPS_object.Cytosol_segmentation(ch, ch2, nuc_s['sort_mask'], nuc_s['sort_mask_bin'])
#plt.imshow(seg['cseg_mask'])
#ch2 = ch2*2**16
ch2 = (ch2/ch2.max())*255
ch2 = np.uint8(ch2)
composite = np.zeros((np.shape(ch2)[0],np.shape(ch2)[1],3),dtype=np.uint8)
bf_mask = dx.binary_frame_mask(ch2, seg['sort_mask_sync'])
bf_mask = np.where(bf_mask == 1, True, False)
c_mask = dx.binary_frame_mask(ch2, seg['cseg_mask'])
c_mask = np.where(c_mask == 1, True, False)
composite[:,:,0] = ch2
composite[:,:,1] = ch2
composite[:,:,2] = ch2
composite[bf_mask > 0, 2] = 255
cseg_mask=seg['cseg_mask']
external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/object_properties_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(
    [
            dbc.Button('Target', id='target', color="success", className="me-1", n_clicks=0, active=True,
                                   style={'font-weight': 'normal'}, size='lg'),
            html.Div(id='Tab_image_display'),
            dcc.Store(id='json_img'),
            dcc.Store(id='json_roi',storage_type='local'),
            html.Div(id='dump',hidden=True),
            html.Div(id='txt_out'),
    ]
)

@app.callback(
    [Output('dump', "children"),
     Output('json_img','data'),
     Output('txt_out','children'),
     Output('json_roi', 'data'),
     ],
    [Input('graph','clickData'),
    State('json_roi', 'data')])

def display_selected_data(clickData,roi):
    if clickData is None:
        return dash.no_update,dash.no_update,dash.no_update
    else:
        #load 3d np array with seed segmentation
        points = clickData['points']
        value = cseg_mask[points[0]['y'],points[0]['x']]
        if roi is None:
            roi = []
            roi.append(value)
        else:
            roi.append(value)
        bf_mask_sel = np.zeros(np.shape(cseg_mask),dtype=np.int32)
        bf_mask_sel[cseg_mask == value] = value
        c_mask = dx.binary_frame_mask_single_point(bf_mask_sel)
        c_mask = np.where(c_mask == 1, True, False)
        composite[c_mask > 0, 1] = 255
        json_object_fig_updata = json.dumps(composite.tolist())
        return json.dumps(clickData, indent=2),json_object_fig_updata, [dbc.Alert(['ROI_{}'.format(i) for i in roi])],roi
#display_mask
@app.callback(
            Output('Tab_image_display', 'children'),
             [Input('target','n_clicks'),
            State('json_img','data')])
def display_image(n,json_img):
    try:
        img_jason = img_as_ubyte(color.gray2rgb(np.array(json.loads(json_img))))
    except:
        img = img_as_ubyte(composite)
        img_input_rgb_pil = Image.fromarray(img)
        fig = px.imshow(img_input_rgb_pil, binary_string=True, binary_backend="jpg", )
        return dcc.Graph(
            id="graph",
            figure=fig)
    else:
        img_input_rgb_pil = Image.fromarray(img_jason)
        fig = px.imshow(img_input_rgb_pil, binary_string=True, binary_backend="jpg", )
        return dcc.Graph(id="graph",figure=fig)


if __name__ == "__main__":
    app.run_server(debug=False)
