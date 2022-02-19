# color.gray2rgb is removed in the current skitimage creates 4 dimension image
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
from dash_extensions import Download
from dash_extensions.snippets import send_data_frame

from utils.controls import controls, controls_nuc, controls_cyto
from utils.Display_composit import image_with_contour, countor_map,row_highlight
from utils import AIPS_functions as af
from utils import AIPS_module as ai
from utils import display_and_xml as dx
from utils.Dash_functions import parse_contents

import pathlib
from app import app
import glob

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
            dbc.Row([
                    dbc.Col([
                    html.Div(id='Tab_image_display'),
                    daq.BooleanSwitch(id='switch_pick_cell',on=True,label="Select cells",labelPosition="top"),
                            ],md=10),
                    dbc.Col([
                        dcc.Store(id='storage_list_table', storage_type='session'),
                        dcc.Dropdown(id='drop_down_tables', multi=True, persistence=True),
                        html.Div([html.Button("Download csv", id="btn"), Download(id="download")])
                    ],md=2),
                ]),
            html.Div(id='dump',hidden=True),
            dcc.Store(id='jason_ch2'),
            dcc.Store(id='json_ch2_gs_rgb'), #3ch
            dcc.Store(id='json_mask_seed'),
            dcc.Store(id='json_mask_target'),
            dcc.Store(id='json_table_prop'),
            dcc.Store(id='json_img'),
            dcc.Store(id='selected_roi_target',storage_type='session'),
            dcc.Store(id='selected_roi_ctrl',storage_type='session'),
            dcc.Store(id='list_image_name',storage_type='session'),
            ])
    ])

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
    nuc_s = AIPS_object.Nucleus_segmentation(img[nuc_sel],rescale_image=True)
    seg = AIPS_object.Cytosol_segmentation(ch, ch2, nuc_s['sort_mask'], nuc_s['sort_mask_bin'],rescale_image=True)
    # segmentation traces the nucleus segmented image based on the
    ch2 = (ch2 / ch2.max()) * 255
    ch2_u8 = np.uint8(ch2)
    # ROI to 32bit
    sort_mask_sync = seg['sort_mask_sync']
    sort_mask_sync = np.array(sort_mask_sync, dtype=np.int32)
    cseg_mask = seg['cseg_mask']
    cseg_mask = np.array(cseg_mask, dtype=np.int32)
    bf_mask = dx.binary_frame_mask(ch2_u8, sort_mask_sync)
    bf_mask = np.where(bf_mask == 1, True, False)
    c_mask = dx.binary_frame_mask(ch2_u8, cseg_mask)
    c_mask = np.where(c_mask == 1, True, False)
    rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
    rgb_input_img[:, :, 0] = ch2_u8
    rgb_input_img[:, :, 1] = ch2_u8
    rgb_input_img[:, :, 2] = ch2_u8
    rgb_input_img[bf_mask > 0, 2] = 255 # 3d grayscale array where green channel is for seed segmentation
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
     Output('selected_roi_target','data'),
     Output('selected_roi_ctrl', 'data')
     ],
    [Input("graph","clickData"),
    Input('jason_ch2', 'data'),
    Input('json_ch2_gs_rgb', 'data'),
    Input('json_mask_seed', 'data'),
    Input('json_mask_target', 'data'),
    State('json_label_state', 'data'),
    State('selected_roi_target','data'),
    State('selected_roi_ctrl', 'data'),
    State('switch_pick_cell','on'),
     ])
    # Input('json_table_prop', 'data')])
def display_selected_data(clickData,_ch2_jason,json_object_ch2_gs_rgb,json_object_mask_seed,json_object_mask_target,label_color,roi_tar,roi_ctrl,on):
    if clickData is None:
        roi_tar = []
        roi_ctrl = []
        return dash.no_update,dash.no_update,roi_tar,roi_ctrl
    else:
        label_color_sel = json.loads(label_color)
        #load 3d np array with seed segmentation
        ch2_rgb = np.array(json.loads(json_object_ch2_gs_rgb))
        # select seed counter
        mask_target = np.array(json.loads(json_object_mask_target))
        points = clickData['points']
        value = mask_target[points[0]['y'],points[0]['x']]
        # build a counter map
        if 'target' in label_color_sel[0]:
            if on:
                roi_tar.append(value)
            else:
                roi_tar.remove(value)
        else:
            if on:
                roi_ctrl.append(value)
            else:
                roi_ctrl.remove(value)
        #update map
        ch2_rgb = countor_map(mask_target, roi_ctrl, roi_tar, ch2_rgb)
        json_object_fig_updata = json.dumps(ch2_rgb.tolist())
        return json.dumps(clickData, indent=2),json_object_fig_updata,roi_tar,roi_ctrl

@app.callback(
            Output('Tab_image_display', 'children'),
            [Input('json_img','data'),
             Input('json_ch2_gs_rgb', 'data')])
def display_image(json_img,json_ch2_gs_rgb):
    try:
        img_jason = img_as_ubyte(np.array(json.loads(json_img)))
    except:
        img = img_as_ubyte(np.array(json.loads(json_ch2_gs_rgb)))
        img_input_rgb_pil = Image.fromarray(img)
        fig = px.imshow(img_input_rgb_pil, binary_string=True, binary_backend="jpg", )
        return dcc.Graph(
            id="graph",
            figure=fig)
    else:
        img_input_rgb_pil = Image.fromarray(img_jason)
        fig = px.imshow(img_input_rgb_pil, binary_string=True, binary_backend="jpg", )
        return dcc.Graph(id="graph",figure=fig)




# Table_display

@app.callback(Output('Tab_table_display', 'children'),
            [Input('json_table_prop', 'data'),
            Input('selected_roi_ctrl','data'),
            Input('selected_roi_target','data'),
            Input('json_label_state', 'data')])
def load_image_and_table(table_prop,roi_ctrl,roi_target,label_color):
    if table_prop is None:
        return dash.no_update
    table = pd.read_json(table_prop,orient='split')
    if roi_ctrl is None and roi_target is None:
        roi_ctrl = []
        roi_target = []
    return  [dbc.Card([
                 dbc.CardBody(
                     dbc.Row(
                         dbc.Col(
                             [
                             dash_table.DataTable(
                                 id="table-line",
                                 columns=[{"name": i, "id": i} for i in table.columns],
                                 data=table.to_dict("records"),
                                 style_data_conditional=row_highlight(roi_ctrl,roi_target),
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
                 ]),
        dbc.Button('Insert', id='inst', color="danger", className="me-1", n_clicks=0, active=True,
                   style={'font-weight': 'bold'}, size='sm'),]

# create a dropdown containing all the names of the files as datatable and add the roi-list
@app.callback([Output('storage_list_table', 'data'),
                Output('drop_down_tables','options'),
                Output('drop_down_tables','value'),
               Output('list_image_name','data')],
                [Input('inst', 'n_clicks'),
                Input('upload-image', 'filename'),
                Input('json_table_prop', 'data'),
                Input('selected_roi_ctrl','data'),
                Input('selected_roi_target','data'),
                Input('storage_list_table', 'data'),
                Input('list_image_name','data')])
def update_dropdown_table_list(n,image,table_prop,roi_ctrl,roi_target,table_sum_input,list_img_name):
    if n == 0:
        return dash.no_update,dash.no_update,dash.no_update,dash.no_update
    if table_prop is None:
        return dash.no_update,dash.no_update,dash.no_update,dash.no_update
    table = pd.read_json(table_prop,orient='split')
    table['class'] = [0]*len(table)
    table['file_name'] = image[0]
    if roi_ctrl is None and roi_target is None:
        roi_ctrl = []
        roi_target = []
    else:
        for roi in roi_ctrl:
            ind = list(table.loc[table['label'].isin([roi])].index)
            table.iloc[ind[0],len(table.columns)-2] = 1
        for roi_t in roi_target:
            ind = list(table.loc[table['label'].isin([roi_t])].index)
            table.iloc[ind[0],len(table.columns)-2] = 2
    if table_sum_input is not None:
        table_sum_out = pd.read_json(table_sum_input,orient='split')
        table_sum = table_sum_out.append(table)
        # remove all duplicate
        table = table_sum.drop_duplicates(subset =['label','file_name'],keep = 'first', inplace = False)
    if list_img_name is None:
        list_img_name = []
    else:
        list_img_name.append(image[0])
        list_img_name = np.unique(list_img_name)
    json_object_table_sum = pd.DataFrame(table).to_json(orient='split')
    return json_object_table_sum, list_img_name, list_img_name,list_img_name

@app.callback([Output("download", "data"),Output("btn", "n_clicks")],
              [Input("btn", "n_clicks"),
               State('storage_list_table', 'data'),
               State('drop_down_tables','value')])
def generate_csv(n,table_sum,list_img_name):
    if n==0:
        return dash.no_update, dash.no_update
    if table_sum is None:
        return dash.no_update, dash.no_update
    table_input = pd.read_json(table_sum, orient='split')
    if len(list_img_name) > 0:
        index = table_input.index[table_input['file_name'].isin(list_img_name)].tolist()
        table = table_input.drop(index=index)
    else:
        table=table_input
    n=0
    return send_data_frame(table.to_csv, filename="some_name.csv"),n




