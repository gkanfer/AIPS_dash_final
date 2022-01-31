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
from utils.Dash_functions import parse_contents

import pathlib
from app import app

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../app_uploaded_files").resolve()
TEMP_PATH = PATH.joinpath("../temp").resolve()

#df = pd.read_csv(os.path.join(PATH,'parameters.csv'))

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
            html.Div(id='Tab_table_display'),
            dcc.Store(id='jason_ch2'),
            dcc.Store(id='json_ch2_gs_rgb'), #3ch
            dcc.Store(id='json_ch2_gs_pil_rgb'),
            dcc.Store(id='json_ch2_seed_gs_pil_rgb'),
            dcc.Store(id='json_ch2_seed_target_gs_pil_rgb'),
            dcc.Store(id='json_mask_seed'),
            dcc.Store(id='json_mask_target'),
            dcc.Store(id='json_table_prop'),
            ])
    ])


# # # loading all the data
@app.callback([
    Output('jason_ch2', 'data'),
    Output('json_ch2_gs_rgb', 'data'),
    Output('json_ch2_gs_pil_rgb', 'data'),
    Output('json_ch2_seed_gs_pil_rgb', 'data'),
    Output('json_ch2_seed_target_gs_pil_rgb', 'data'),
    Output('json_mask_seed','data'),
    Output('json_mask_target','data'),
    Output('json_table_prop','data')],
   [Input('upload-image', 'filename'),
    State('act_ch', 'value'),
    State('high_pass', 'value'),
    State('low_pass', 'value'),
    State('block_size','value'),
    State('offset','value'),
    State('offset_store','data'),
    State('rmv_object_nuc','value'),
    State('block_size_cyto', 'value'),
    State('offset_cyto','value'),
    State('offset_cyto_store', 'data'),
    State('global_ther', 'value'),
    State('rmv_object_cyto', 'value'),
    State('rmv_object_cyto_small', 'value')
     ])
def Generate_segmentation_and_table(image,n,pram,cont,channel,high,low,bs,os,osd,ron,bsc,osc,oscd,gt,roc,rocs):
    '''
    Genrate
    3 channel grayscale target PIL RGB
    3 channel grayscale target PIL RGB image with seed segment
    3 channel grayscale target PIL RGB image with seed and target segment
    complete feture table
    32int seed mask
    32int target mask
    '''
    #test wheter paramters are from csv file
    if osd is None:
        os=os
    else:
        os=osd
    if oscd is None:
        osc=osc
    else:
        osc=oscd
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
    ch = img[nuc_sel]
    ch2 = img[cyt_sel]
    nuc_s = AIPS_object.Nucleus_segmentation(img[nuc_sel])
    seg = AIPS_object.Cytosol_segmentation(ch, ch2, nuc_s['sort_mask'], nuc_s['sort_mask_bin'])
    # segmentation traces the nucleus segmented image based on the
    ch2 = (ch2 / ch2.max()) * 255
    ch2_u8 = np.uint8(ch2)
    bf_mask = dx.binary_frame_mask(ch2_u8, seg['sort_mask_sync'])
    bf_mask = np.where(bf_mask == 1, True, False)
    c_mask = dx.binary_frame_mask(ch2_u8, seg['cseg_mask'])
    c_mask = np.where(c_mask == 1, True, False)
    rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
    rgb_input_img[:, :, 0] = ch2_u8
    rgb_input_img[:, :, 1] = ch2_u8
    rgb_input_img[:, :, 2] = ch2_u8
    img_input_rgb_pil = Image.fromarray(rgb_input_img) # 3 channel grayscale no segmentation
    composite = np.zeros((np.shape(ch2)[0], np.shape(ch2)[1], 3), dtype=np.uint8)
    composite[c_mask > 0, 0] = 255
    composite[bf_mask > 0, 1] = 255
    composite[:, :, 2] = ch2
    img_nuc_cyto_rgb_pil = Image.fromarray(composite) #3 channel grayscale with seed and target segmented
    composite = np.zeros((np.shape(ch2)[0], np.shape(ch2)[1], 3), dtype=np.uint8)
    composite[c_mask > 0, 0] = 255
    composite[:, :, 1] = ch2_u8
    composite[:, :, 2] = ch2_u8
    img_cyto_rgb_pil = Image.fromarray(composite) #3 channel grayscale with seed segmented
    cseg_mask = seg['cseg_mask']
    #label_array = nuc_s['sort_mask']
    prop_names = [
        "label",
        "area",
        "area_bbox",
        "area_convex",
        "area_filled",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "equivalent_diameter_area",
        "euler_number",
        "extent",
        "feret_diameter_max",
        "image_intensity",
        "inertia_tensor",
        "inertia_tensor_eigvals",
        "intensity_max",
        "intensity_mean",
        "intensity_min",
        "moments",
        "moments_central",
        "moments_hu",
        "moments_normalized",
        "moments_weighted",
        "moments_weighted_central",
        "moments_weighted_hu",
        "moments_weighted_normalized",
        "orientation",
        "perimeter",
        "perimeter_crofton",
        "slice",
        "solidity"
    ]
    table_prop = measure.regionprops_table(
        cseg_mask, intensity_image=rgb_input_img, properties=prop_names
    )
    json_object_ch2 = json.dumps(ch2.tolist())
    json_object_ch2_gs_rgb = json.dumps(rgb_input_img.tolist())
    json_object_ch2_gs_pil_rgb = json.dumps(img_input_rgb_pil.tolist())
    json_object_ch2_seed_gs_pil_rgb = json.dumps(img_cyto_rgb_pil.tolist())
    json_object_ch2_seed_target_gs_pil_rgb = json.dumps(img_nuc_cyto_rgb_pil.tolist())
    json_object_mask_seed = json.dumps(seg['sort_mask_sync'].tolist())
    json_object_mask_target = json.dumps(seg['cseg_mask'].tolist())
    json_object_table_prop = table_prop.to_json(orient='split')
    return json_object_ch2,json_object_ch2_gs_rgb ,json_object_ch2_gs_pil_rgb, json_object_ch2_seed_gs_pil_rgb, json_object_ch2_seed_target_gs_pil_rgb\
            ,json_object_mask_seed,json_object_mask_target,json_object_table_prop

#
# ### load image and table side by side
# @app.callback(
#             [Output('Tab_image_display', 'children'),
#              Output('Tab_table_display', 'children')],
#             [Input('json_object_ch2', 'data'),
#             Input('json_object_ch2_gs_rgb', 'data'),
#             Input('json_object_ch2_gs_pil_rgb', 'data'),
#             Input('json_object_ch2_seed_gs_pil_rgb', 'data'),
#             Input('json_object_ch2_seed_target_gs_pil_rgb', 'data'),
#             Input('json_object_mask_seed', 'data'),
#             Input('json_object_mask_target', 'data'),
#             Input('json_object_table_prop', 'data')])
# def load_image_and_table(json_object_ch2_gs_rgb,json_object_ch2_gs_pil_rgb, json_object_ch2_seed_gs_pil_rgb,
#                          json_object_ch2_seed_target_gs_pil_rgb,json_object_mask_seed,json_object_mask_target, json_object_table_prop):
#     ch2_rgb = json.loads(json_object_ch2_gs_rgb)
#     ch2_rgb_pil = json.loads(json_object_ch2_gs_pil_rgb)
#     ch2_seed_gs_pil_rgb = json.loads(json_object_ch2_seed_gs_pil_rgb)
#     ch2_seed_target_gs_pil_rgb = json.loads(json_object_ch2_seed_target_gs_pil_rgb)
#     mask_seed = json.loads(json_object_mask_seed)
#     mask_target = json.loads(json_object_mask_target)
#     table_prop = pd.read_json(json_object_table_prop,orient='split')
#     table = table_prop.iloc[:,[0,1,2,4,10,14]]
#     prop_names = []
#     [prop_names.append(str(i)) for i in table.columns]
#     columns = [
#         {"name": label_name, "id": label_name, "selectable": True}
#         if precision is None
#         else {
#             "name": label_name,
#             "id": label_name,
#             "type": "numeric",
#             "selectable": True,
#         }
#         for label_name, precision in zip(prop_names, (None, None, 4, 4, None, 3))]
#     initial_columns = ["label", "area"]
#     return [
#             dbc.CardHeader(html.H2("Explore seed properties - SVM", style={'text-align': 'center'})),
#             dbc.CardBody(
#                 dbc.Row(
#                     dbc.Col(
#                         dcc.Graph(
#                             id="graph",
#                             figure=image_with_contour(
#                                 ch2_seed_gs_pil_rgb,
#                                 mask_target,
#                                 table,
#                                 initial_columns,
#                                 color_column="area",
#                             ),
#                         ),
#                     )
#                 )
#             ),
#         dbc.CardBody(
#             dbc.Row(
#                 dbc.Col(
#                     [
#                         dash_table.DataTable(
#                             id="table-line",
#                             columns=columns,
#                             data=table.to_dict("records"),
#                             tooltip_header={
#                                 col: "Select columns with the checkbox to include them in the hover info of the image."
#                                 for col in table.columns
#                             },
#                             style_header={
#                                 "textDecoration": "underline",
#                                 "textDecorationStyle": "dotted",
#                             },
#                             tooltip_delay=0,
#                             tooltip_duration=None,
#                             filter_action="native",
#                             row_deletable=True,
#                             column_selectable="multi",
#                             selected_columns=initial_columns,
#                             style_table={"overflowY": "scroll"},
#                             fixed_rows={"headers": False, "data": 0},
#                             style_cell={"width": "85px"},
#                             page_size=10,
#                         ),
#                         html.Div(id="row", hidden=True, children=None),
#                     ]
#                 )
#             )),
#              ]

