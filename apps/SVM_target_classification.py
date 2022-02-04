# use my interpretation of display segmentation
import dash_daq as daq
import json
import dash
from dash import callback_context
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
            dbc.Button('Target', id='target', color="success", className="me-1", n_clicks=0, active=True,
                       style={'font-weight': 'normal'}, size='lg'),
            dbc.Button('Control', id='control', color="danger", className="me-1", n_clicks=0, active=True,
                       style={'font-weight': 'bold'}, size='sm'),
            html.Div(id='json_label_state'),
            html.Div(id='Tab_image_display'),
            html.Div(id='dump',hidden=True),
            dcc.Store(id='jason_ch2'),
            dcc.Store(id='json_ch2_gs_rgb'), #3ch
            dcc.Store(id='json_mask_seed'),
            dcc.Store(id='json_mask_target'),
            dcc.Store(id='json_table_prop'),
            dcc.Store(id='json_img'),
            dcc.Store(id = 'json_selected_roi'),
            ])
    ])

#
# # # # loading all the data
@app.callback([
    Output('jason_ch2', 'data'),
    Output('json_ch2_gs_rgb', 'data'),
    Output('json_mask_seed','data'),
    Output('json_mask_target','data'),
    Output('json_table_prop','data')],
   [Input('upload-image', 'filename'),
    State('act_ch', 'value'),
    State('block_size','value'),
    State('offset','value'),
    State('offset_store','data'),
    State('rmv_object_nuc','value'),
    State('block_size_cyto', 'value'),
    State('offset_cyto','value'),
    State('offset_cyto_store', 'data'),
    State('global_ther', 'value'),
    State('rmv_object_cyto', 'value'),
    State('rmv_object_cyto_small', 'value')])
def Generate_segmentation_and_table(image,channel,bs,os,osd,ron,bsc,osc,oscd,gt,roc,rocs):
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
    rgb_input_img[bf_mask > 0, 1] = 255 # 3d grayscale array where green channel is for seed segmentation
    cseg_mask = seg['cseg_mask']
    #label_array = nuc_s['sort_mask']
    prop_names = [
        "label",
        "area",
        "eccentricity",
        "euler_number",
        "extent",
        "feret_diameter_max",
        "inertia_tensor",
        "inertia_tensor_eigvals",
        "moments",
        "moments_central",
        "moments_hu",
        "moments_normalized",
        "orientation",
        "perimeter",
        "perimeter_crofton",
        # "slice",
        "solidity"
    ]
    table_prop = measure.regionprops_table(
        cseg_mask, intensity_image=rgb_input_img, properties=prop_names
    )
    json_object_ch2 = json.dumps(ch2.tolist())
    json_object_ch2_seed_gs_rgb = json.dumps(rgb_input_img.tolist())
    json_object_mask_seed = json.dumps(seg['sort_mask_sync'].tolist())
    json_object_mask_target = json.dumps(seg['cseg_mask'].tolist())
    json_object_table_prop = pd.DataFrame(table_prop).to_json(orient='split')
    return json_object_ch2,json_object_ch2_seed_gs_rgb ,json_object_mask_seed,json_object_mask_target,json_object_table_prop

#
# ### load image and table side by side
# generate selected map
@app.callback(
    [Output("dump", "children"),
     Output('json_img','data'),
     Output('json_selected_roi','data')],
    [Input("graph","clickData"),
    Input('jason_ch2', 'data'),
    Input('json_ch2_gs_rgb', 'data'),
    Input('json_mask_seed', 'data'),
    Input('json_mask_target', 'data')])
    # Input('json_table_prop', 'data')])
def display_selected_data(clickData,_ch2_jason,json_object_ch2_gs_rgb,json_object_mask_seed,json_object_mask_target):
    if clickData is None:
        return dash.no_update,dash.no_update,dash.no_update
    else:
        #load 3d np array with seed segmentation
        ch2_rgb = np.array(json.loads(json_object_ch2_gs_rgb))
        # select seed counter
        mask_target = np.array(json.loads(json_object_mask_target))
        points = clickData['points']
        value = mask_target[points[0]['y'],points[0]['x']]
        bf_mask_sel = np.zeros(np.shape(mask_target),dtype=np.int32)
        bf_mask_sel[mask_target == value] = value
        c_mask = dx.binary_frame_mask_single_point(bf_mask_sel)
        c_mask = np.where(c_mask == 1, True, False)
        ch2_rgb[c_mask > 0, 2] = 255
        json_object_fig_updata = json.dumps(ch2_rgb.tolist())
        roi_sel = json.dumps(value.tolist())
        return json.dumps(clickData, indent=2),json_object_fig_updata,roi_sel

@app.callback(
            Output('Tab_image_display', 'children'),
            [Input('json_img','data'),
             Input('json_ch2_gs_rgb', 'data')])
def display_image(json_img,json_ch2_gs_rgb):
    try:
        img_jason = img_as_ubyte(color.gray2rgb(np.array(json.loads(json_img))))
    except:
        img = img_as_ubyte(color.gray2rgb(np.array(json.loads(json_ch2_gs_rgb))))
        img_input_rgb_pil = Image.fromarray(img)
        fig = px.imshow(img_input_rgb_pil, binary_string=True, binary_backend="jpg", )
        return dcc.Graph(
            id="graph",
            figure=fig)
    else:
        img_input_rgb_pil = Image.fromarray(img_jason)
        fig = px.imshow(img_input_rgb_pil, binary_string=True, binary_backend="jpg", )
        return dcc.Graph(id="graph",figure=fig)

# Selected label

@app.callback(
    [Output('target', 'style'),
     Output('control', 'style'),
    Output('target', 'size'),
     Output('control', 'size'),
     Output('json_label_state','data')],
    [Input('target', 'n_clicks'),
    Input('control', 'n_clicks')])
def displayClick(targt_btn, ctrl_btn):
    if targt_btn and ctrl_btn is None:
        return dash.no_update, dash.no_update, dash.no_update,dash.no_update,dash.no_update
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    jason_label = json.dumps([changed_id])
    if 'target' in changed_id:
        return {'font-weight': 'bold'},{'font-weight': 'normal'},"lg","sm",jason_label
    else:
        return {'font-weight': 'normal'},{'font-weight': 'bold'},"sm","lg",jason_label

@app.callback(Output('Tab_table_display', 'children'),
            [Input('json_table_prop', 'data'),
            Input('json_selected_roi','data'),
            Input('json_label_state', 'data')])
def load_image_and_table(table_prop,roi,label_color):
    table = pd.read_json(table_prop,orient='split')
    if roi is None or label_color is None:
        return  [dbc.Card([
                 dbc.CardBody(
                     dbc.Row(
                         dbc.Col(
                             [
                             dash_table.DataTable(
                                 id="table-line",
                                 columns=columns,
                                 data=table.to_dict("records"),
                                 style_header={
                                     "textDecoration": "underline",
                                     "textDecorationStyle": "dotted",
                                 },
                                 tooltip_delay=0,
                                 tooltip_duration=None,
                                 filter_action="native",
                                 row_deletable=True,
                                 column_selectable="multi",
                                 style_table={"overflowX": "scroll"},
                                 fixed_rows={"headers": False, "data": 0},
                                 style_cell={"width": "85px"},
                                 page_size=10,
                             ),
                         ]
                     )
                 )),
                 ])]
    else:
        roi = json.loads(roi)
        x = table.loc[table['label']==int(roi),'label']
        label_color_sel = json.loads(label_color)
        if 'target' in label_color_sel[0]:
            color = '#1ABA19'
        else:
            color = '#F31515'
        return  [dbc.Card([
                         dbc.CardBody(
                             dbc.Row(
                                 dbc.Col(
                                     [
                                         dash_table.DataTable(
                                             id="table-line",
                                             # columns=columns,
                                             data=table.to_dict("records"),
                                             columns= [{"name": i, "id": i}
                                                       for i in table.columns],
                                             style_data_conditional=[
                                                 {
                                                     'if': {'filter_query': '{{label}} = {}'.format(int(x))},
                                                     'backgroundColor': '{}'.format(color),
                                                     'color': 'white'
                                                 }
                                             ],
                                             tooltip_delay=0,
                                             tooltip_duration=None,
                                             filter_action="native",
                                             row_deletable=True,
                                             column_selectable="multi",
                                             style_table={"overflowX": "scroll"},
                                             fixed_rows={"headers": False, "data": 0},
                                             style_cell={"width": "85px"},
                                             row_selectable="multi",
                                             page_size=10,
                                         ),
                                     ]
                                 )
                             )),
                             ])]





#
# @app.callback(Output('Tab_table_display', 'children'),
#             [Input('json_table_prop', 'data'),
#             Input('json_selected_roi','data'),
#             Input('json_label_state', 'data')])
# def load_image_and_table(table_prop,roi,label_color):
#     table = pd.read_json(table_prop,orient='split')
#     if roi is None or label_color is None:
#          return  [dbc.Card([
#                  dbc.CardBody(
#                      dbc.Row(
#                          dbc.Col(
#                              [
#                                  dash_table.DataTable(
#                                      id="table-line",
#                                      # columns=columns,
#                                      data=table.to_dict("records"),
#                                      columns= [{"name": i, "id": i}
#                                                for i in table.columns],
#                                      tooltip_delay=0,
#                                      tooltip_duration=None,
#                                      filter_action="native",
#                                      row_deletable=True,
#                                      column_selectable="multi",
#                                      style_table={"overflowX": "scroll"},
#                                      fixed_rows={"headers": False, "data": 0},
#                                      style_cell={"width": "85px"},
#                                      row_selectable="multi",
#                                      style_data_conditional={"filter_query":'0',"backgroundColor": "yellow",},
#                                      page_size=10,
#                                  ),
#                              ]
#                          )
#                      )),
#                      ])]
#     else:
#         roi = json.loads(roi)
#         x = table.loc[table['label']==int(roi),'label']
#         label_color_sel = json.loads(label_color)
#         if 'target' in label_color_sel:
#             color = '#1ABA19'
#         else:
#             color = '#F31515'
#         return  [dbc.Card([
#                          dbc.CardBody(
#                              dbc.Row(
#                                  dbc.Col(
#                                      [
#                                          dash_table.DataTable(
#                                              id="table-line",
#                                              # columns=columns,
#                                              data=table.to_dict("records"),
#                                              columns= [{"name": i, "id": i}
#                                                        for i in table.columns],
#                                              style_data_conditional=[
#                                                  {
#                                                      'if': {'filter_query': '{{label}} = {}'.format(int(x))},
#                                                      'backgroundColor': '{}'.format(color),
#                                                      'color': 'white'
#                                                  }
#                                              ],
#                                              tooltip_delay=0,
#                                              tooltip_duration=None,
#                                              filter_action="native",
#                                              row_deletable=True,
#                                              column_selectable="multi",
#                                              style_table={"overflowX": "scroll"},
#                                              fixed_rows={"headers": False, "data": 0},
#                                              style_cell={"width": "85px"},
#                                              row_selectable="multi",
#                                              page_size=10,
#                                          ),
#                                      ]
#                                  )
#                              )),
#                              ])]
#




