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
seg = AIPS_object.Cytosol_segmentation( ch,ch2,nuc_s['sort_mask'],nuc_s['sort_mask_bin'])
cseg_mask = seg['cseg_mask']

# generate PIL RGB image ready for fig
ch2 = (ch2 / ch2.max()) * 255
ch2_u8 = np.uint8(ch2)
rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
rgb_input_img[:, :, 0] = ch2_u8
rgb_input_img[:, :, 1] = ch2_u8
rgb_input_img[:, :, 2] = ch2_u8
img = img_as_ubyte(color.gray2rgb(rgb_input_img))
img_input_rgb_pil = Image.fromarray(img)
# generate complete mask
cseg_mask = seg['cseg_mask']
#mask_target = np.pad(cseg_mask, (1,), "constant", constant_values=(0,)) # for the function
nseg_mask=seg['sort_mask_sync']

# generate table
prop_names = [
    "label",
    "area",
    "perimeter",
    "eccentricity",
    "euler_number",
    "mean_intensity",
]
table_prop = measure.regionprops_table(
    cseg_mask, intensity_image=rgb_input_img, properties=prop_names
)
columns = [
    {"name": prop_names, "id": prop_names, "selectable": True}]
# add figure information
#fig = px.imshow(img_input_rgb_pil, binary_string=True, binary_backend="jpg", )

config = {
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ]
}

# Set up the app
external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/object_properties_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        dbc.CardBody(
            dbc.Row([
                dbc.Col(
                    html.Div(id='dump'),
                ),
                    dbc.Col(
                    html.Button('Submit', id='submit-val', n_clicks=0),
                    ),
                dbc.Col(
                    dcc.Graph(
                        id="graph",
                            config=config
                        ),
                    ),
                ]),
        ),
        dcc.Store(id='for_fig'),
        dbc.Card([
                dbc.CardBody(
                    dbc.Row(
                        dbc.Col(
                            [
                                dash_table.DataTable(
                                    id="table-line",
                                    columns=columns,
                                    # data=table.to_dict("records"),
                                    # tooltip_header={
                                    #     col: "Select columns with the checkbox to include them in the hover info of the image."
                                    #     for col in table.columns
                                    # },
                                    style_header={
                                        "textDecoration": "underline",
                                        "textDecorationStyle": "dotted",
                                    },
                                    tooltip_delay=0,
                                    tooltip_duration=None,
                                    filter_action="native",
                                    row_deletable=True,
                                    column_selectable="multi",
                                    # selected_columns=initial_columns,
                                    style_table={"overflowY": "scroll"},
                                    fixed_rows={"headers": False, "data": 0},
                                    style_cell={"width": "85px"},
                                    page_size=10,
                                ),
                            ]
                        )
                    )),
                ])
        ])


def InteractiveImage(id):
    fig = px.imshow(img_input_rgb_pil, binary_string=True, binary_backend="jpg", )
    return dcc.Graph(
        id=id,
        figure=fig)



@app.callback(
    [Output("dump", "children"),
     Output('for_fig','data')],
    Input("graph","clickData"))
def display_selected_data(clickData):
    if clickData is None:
        return dash.no_update,dash.no_update
    # points = clickData['points']
    # value = cseg_mask[points[0]['x'],points[0]['y']]
    # bf_mask_sel = np.zeros(np.shape(cseg_mask),dtype=np.int32)
    # bf_mask_sel[cseg_mask == value] = value
    # c_mask = dx.binary_frame_mask_single_point(bf_mask_sel)
    # c_mask = np.where(c_mask == 1, True, False)
    n_mask = dx.binary_frame_mask(ch2, nseg_mask)
    n_mask = np.where(n_mask == 1, True, False)
    # img = img_as_ubyte(color.gray2rgb(rgb_input_img))
    # img_input_rgb_pil = Image.fromarray(img)
    rgb_input_img[n_mask > 0, 1] = 255
    # rgb_input_img[c_mask > 0, 2] = 255
    img = img_as_ubyte(color.gray2rgb(rgb_input_img))
    json_object_fig_updata = json.dumps(img.tolist())
    return json.dumps(clickData, indent=2),json_object_fig_updata


@app.callback(
    Output("graph", "figure"),
    [Input('submit-val','n_clicks'),
    Input('for_fig','data')])
def click(n,for_fig):
    if n is None:
        return dash.no_update
    try:
        #img_jason = np.array(json.loads(for_fig))
        img_jason = img_as_ubyte(color.gray2rgb(np.array(json.loads(for_fig))))
    except:
        img = img_as_ubyte(color.gray2rgb(rgb_input_img))
        img_input_rgb_pil = Image.fromarray(img)
        fig = px.imshow(img_input_rgb_pil, binary_string=True, binary_backend="jpg", )
        return fig
    else:

        img_input_rgb_pil = Image.fromarray(img_jason)
        fig = px.imshow(img_input_rgb_pil, binary_string=True, binary_backend="jpg", )
        return fig
    # #fig = px.imshow(img_input_rgb_pil, binary_string=True, binary_backend="jpg", )
    # #points = clickData['points']
    # # value = cseg_mask[points[0]['x'],points[0]['y']]
    # # bf_mask_sel = np.zeros(np.shape(cseg_mask),dtype=np.int32)
    # # bf_mask_sel[cseg_mask == value] = value
    # # c_mask = dx.binary_frame_mask_single_point(bf_mask_sel)
    # # c_mask = np.where(c_mask == 1, True, False)
    # # n_mask = dx.binary_frame_mask(ch2,nseg_mask)
    # # n_mask = np.where(n_mask == 1, True, False)
    # # img = img_as_ubyte(color.gray2rgb(rgb_input_img))
    # # img_input_rgb_pil = Image.fromarray(img)
    # rgb_input_img[n_mask > 0, 1] = 255
    # # rgb_input_img[c_mask > 0, 2] = 255
    # img = img_as_ubyte(color.gray2rgb(rgb_input_img))
    # img_input_rgb_pil = Image.fromarray(img)
    # fig = px.imshow(img_input_rgb_pil, binary_string=True, binary_backend="jpg", )
    # return fig


if __name__ == '__main__':
    app.run_server()