'''
# Update branch
git add .
git commit -m "03-09-2022 change marks value"
git branch -m server
git push origin -u server
'''
import dash_labs as dl
import json
import dash
from dash import ALL, callback_context
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Dash, Output, Input, State, ServersideOutput
import tifffile as tfi
import os
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from PIL import Image
import plotly.express as px
import pathlib
import base64
import pandas as pd
import io
from io import BytesIO
import re
from utils.controls import controls, controls_nuc, controls_cyto, upload_parm
from utils.Dash_functions import parse_contents
from utils import AIPS_functions as af
from utils import AIPS_module as ai
import pathlib
import dash_canvas
from dash_canvas.components import image_upload_zone
from dash_canvas.utils import (
    image_string_to_PILImage,
    array_to_data_url,
    parse_jsonstring_line,
    brightness_adjust,
    contrast_adjust,
)
UPLOAD_DIRECTORY = "/app_uploaded_files"

# style_nav = {'background-color':'#01183A',
#           'max-width': '550px',
#           'width': '100%'}

app = dash.Dash(
    __name__,
    plugins=[dl.plugins.pages],
    external_stylesheets=[dbc.themes.JOURNAL, dbc.icons.FONT_AWESOME],
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=True
)

nav_bar = dbc.Nav(
    [
    dbc.NavItem(dbc.NavLink('display',id='dis', href=dash.page_registry['pages.Image_display']['path'], active=True,disabled=False,class_name='page-link')),
    html.Div(className='separator'),
    dbc.NavItem(dbc.NavLink('Nucleus',id='nuc', href=dash.page_registry['pages.Nucleus_segmentation']['path'], active=False,disabled=True,class_name='page-link')),
    html.Div(className='separator'),
    dbc.NavItem(dbc.NavLink('Target',id='tar', href=dash.page_registry['pages.target_segmentation']['path'], active=False,disabled=True,class_name='page-link')),
    html.Div(className='separator'),
    dbc.NavItem(dbc.NavLink('Download parameters',id='down', href=dash.page_registry['pages.download_parametars']['path'],active=False, disabled=True,class_name='page-link')),
    dbc.NavItem(children=[
                dbc.DropdownMenu(
                        children = [
                        dbc.DropdownMenuItem("Nucleus count predict", href=dash.page_registry['pages.modules.Nuclues_count_predict']['path']),
                        dbc.DropdownMenuItem("SVM target classification", href=dash.page_registry['pages.modules.SVM_target_classification']['path']),
                        ],
                        label="Modules")
                        ]),
                    ],
        pills=True,
        fill=False,
        justified=True,
        navbar=True,
        #className="page-link"
    )

app.layout = dbc.Container(
    [
        html.H1("Optical Pooled Cell Sorting Platform",className='header'),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                 dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),className='file_uploader',multiple=True
                    ),
                html.Button('Submit', id='submit-val', n_clicks=0),
                dbc.Accordion(
                            [
                                dbc.AccordionItem(children=
                                [
                        controls,
                                ],title='Image configuration'),
                                dbc.AccordionItem(children=
                                [
                        controls_nuc,
                                ],title='Seed segmentation config'),
                                dbc.AccordionItem(children=
                                [
                        controls_cyto,
                                ], title='Target segmentation config'),
                                dbc.AccordionItem(children=
                                [
                        upload_parm,

                                ], title='Update parameters'),
                            ], start_collapsed=True),
                html.Br(),
                html.Div(id='Tab_table_display'),
            ], width={"size": 4}),
            dbc.Col([
                nav_bar,
                html.Div(id='Tab_slice'),
                dl.plugins.page_container,
                html.Div(id='ch_holder', children=[]),
                html.Div(id='ch2_holder',children=[]),
                dcc.Store(id='slice_selc',data=None),
                dcc.Store(id='ch_slice',data=None),
                dcc.Store(id='ch2_slice',data=None),
                dcc.Store(id='json_img_ch',data=None),
                dcc.Store(id='json_img_ch2',data=None),
                dcc.Store(id='json_react', data=None), # rectangle for memory reduction
                dcc.Store(id='offset_store',data=None),
                dcc.Store(id='offset_cyto_store',data=None),
                dcc.Store(id='slider-memory-scale', data=None),
                html.Div(id="test-image-name",hidden=True),
                dcc.Interval(id = 'interval',interval=1000,max_intervals=2,disabled=True)
            ]),
            ])],
    fluid=True)


@app.callback(
    [Output('act_ch', 'value'),
    Output('block_size', 'value'),
    Output('offset_store', 'data'),
    Output('rmv_object_nuc', 'value'),
    Output('block_size_cyto', 'value'),
    Output('offset_cyto_store', 'data'),
    Output('global_ther', 'value'),
    Output('rmv_object_cyto', 'value'),
    Output('rmv_object_cyto_small', 'value'),
    Output('set-val','n_clicks'),
    Output('set-val-cyto','n_clicks'),
     ],
    [Input('submit-parameters', 'n_clicks'),
     State('upload-csv', 'filename'),
     State('upload-csv', 'contents')])
def Load_image_parameters_table(n,pram,cont):
    if n < 1:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,\
               dash.no_update,dash.no_update, dash.no_update, dash.no_update,dash.no_update,dash.no_update
    parameters = parse_contents(cont,pram)
    channel = int(parameters['act_ch'][0])
    bs = parameters['block_size'][0]
    os = parameters['offset'][0]
    ron = parameters['rmv_object_nuc'][0]
    bsc = parameters['block_size_cyto'][0]
    osc = parameters['offset_cyto'][0]
    gt = parameters['global_ther'][0]
    roc = parameters['rmv_object_cyto'][0]
    rocs = parameters['rmv_object_cyto_small'][0]
    #mem = parameters['memory_reduction'][0] #(1,2,3,4)
    set_nuc=1
    set_cyt=1
    return channel,bs,os,ron,bsc,osc,gt,roc,rocs,set_nuc,set_cyt


@app.callback(
    [ServersideOutput('json_img_ch', 'data'),
    ServersideOutput('json_img_ch2', 'data')],
    [Input('submit-val', 'n_clicks'),
    State('upload-image', 'filename'),
    State('upload-image', 'contents'),
    Input('act_ch', 'value'),
    Input('json_react','data'),
    State('downscale_image_on','on'),
     ],memoize=True)
def Load_image(n,image,cont,channel_sel,react,downscale):
    '''
    react: reactangle from draw compnante of user
    '''
    w = None
    h = None
    if n == 0:
        return dash.no_update,dash.no_update
    content_string = cont[0].split('data:image/tiff;base64,')[1]
    decoded = base64.b64decode(content_string)
    pixels = tfi.imread(io.BytesIO(decoded))
    pixels_float = pixels.astype('float64')
    if downscale:
        if np.shape(pixels_float)[1] > np.shape(pixels_float)[2]:
            w = int(np.shape(pixels_float)[1] / (np.shape(pixels_float)[1] / 500))
            h = int(np.shape(pixels_float)[2] / (np.shape(pixels_float)[1] / 500))
        else:
            w = int(np.shape(pixels_float)[1] / (np.shape(pixels_float)[2] / 500))
            h = int(np.shape(pixels_float)[2] / (np.shape(pixels_float)[2] / 500))
    img = pixels_float / 65535.000
    if channel_sel == 1:
        ch_ = img[0,:,:]
        ch2_ = img[1,:,:]
    else:
        ch_ = img[1,:,:]
        ch2_ = img[0,:,:]
    if react is not None:
        y0, y1, x0, x1 = react
        ch_ = ch_[y0:y1, x0:x1]
        ch2_ = ch2_[y0:y1, x0:x1]
    if w is not None:
        ch_ = resize(ch_,(h,w))
        ch2_ = resize(ch2_,(h,w))
    json_object_img_ch = ch_
    json_object_img_ch2 = ch2_
    return json_object_img_ch,json_object_img_ch2


@app.callback(Output('graduated-bar', 'value'),
              Input('graduated-bar-slider', 'value'))
def update_bar(bar_slider):
    return bar_slider

@app.callback(Output('graduated-bar-nuc-zoom', 'value'),
              Input('graduated-bar-slider-nuc-zoom', 'value'))
def update_bar2(bar_slider_zoom):
    return bar_slider_zoom


@app.callback(
    [Output('offset', 'min'),
     Output('offset', 'max'),
     Output('offset', 'marks'),
     Output('offset', 'value'),
     Output('offset', 'step')],
    [Input('set-val','n_clicks'),
    Input('graduated-bar-slider-nuc-zoom', 'value'),
    State('json_img_ch', 'data'),
     State('Auto-nuc', 'on'),
     State('offset', 'value'),
     State('graduated-bar-slider', 'value'),
     State('upload-image', 'filename'),
     State('act_ch', 'value'),
     State('submit-parameters', 'n_clicks'),
     State('upload-csv', 'filename'),
     State('upload-csv', 'contents'),
     ])
def Updat_offset(set_n,bar_zoom,ch,au,offset_input,bar_ind,image_input,channel_sel,n_parm,pram,cont):
    if au:
        ch_ = np.array(ch)
        med_nuc = np.median(ch_) / 400
        norm = np.random.normal(med_nuc, 0.001*bar_ind, 100)
        offset_pred = []
        for norm_ in norm:
            AIPS_object = ai.Segment_over_seed(Image_name=image_input[0], path=UPLOAD_DIRECTORY, rmv_object_nuc=0.9,
                                               block_size=59, offset=norm_,
                                               block_size_cyto=9, offset_cyto=0.0004, global_ther=0.4,
                                               rmv_object_cyto=0.99, rmv_object_cyto_small=0.9, remove_border=True)
            nuc_s = AIPS_object.Nucleus_segmentation(ch_, inv=False)
            offset_pred = norm_
            len_table = len(nuc_s['tabale_init'])
            if len_table > 3:
                break
        norm = np.random.normal(offset_pred, 0.001, 100)
        min_val = round(np.min(norm), 4)
        max_val = round(np.max(norm), 4)
        steps = (np.max(norm) - np.min(norm)) / bar_zoom
        value_marks = {i: i for i in [min_val, max_val]}
        return [min_val, max_val, value_marks, offset_pred,steps]
    else:
        if n_parm > 0:
            parameters = parse_contents(cont, pram)
            os = parameters['offset'][0]
            norm = np.random.normal(os, 0.001, 100)
            min_val = round(np.min(norm), 4)
            max_val = round(np.max(norm), 4)
            steps = (np.max(os) - np.min(os)) / bar_zoom
            value_marks = {i: i for i in [min_val, max_val]}
            offset_pred = os
        else:
            min_val = 0.001
            max_val = 0.8
            value_marks = {i: i for i in [0.001, 0.8]}
            offset_pred = offset_input
            steps = 0.001
        return [min_val, max_val, value_marks, offset_pred,steps]

'''
    Initiate offset parameter prediction for Cytosol segmentation
'''

@app.callback(Output('graduated-bar-cyto', 'value'),
              Input('graduated-bar-slider-cyto', 'value'))
def update_bar_cyto(bar_slider_cyto):
    return bar_slider_cyto

@app.callback(Output('graduated-bar-cyto-zoom', 'value'),
              Input('graduated-bar-slider-cyto-zoom', 'value'))
def update_bar3(bar_slider_cyto_zoom):
    return bar_slider_cyto_zoom


@app.callback(
    [Output('offset_cyto', 'min'),
     Output('offset_cyto', 'max'),
     Output('offset_cyto', 'marks'),
     Output('offset_cyto', 'value'),
     Output('offset_cyto', 'step')],
    [Input('set-val-cyto','n_clicks'),
     Input('graduated-bar-slider-cyto-zoom', 'value'),
    State('json_img_ch2', 'data'),
    State('Auto-cyto', 'on'),
    State('offset_cyto', 'value'),
    State('graduated-bar-cyto', 'value'),
    State('upload-image', 'filename'),
    State('act_ch', 'value'),
    State('submit-parameters', 'n_clicks'),
    State('upload-csv', 'filename'),
    State('upload-csv', 'contents'),
     ])
def Updat_offset_cyto(set_n,bar_zoom_cyto,ch2,au,offset_input,bar_ind,image_input,channel_sel,n_parm,pram,cont):
    if au:
        ch2_ = np.array(ch2)
        med_nuc = np.median(ch2_) / 400
        norm = np.random.normal(med_nuc, 0.001*bar_ind, 100)
        offset_pred = []
        for norm_ in norm:
            AIPS_object = ai.Segment_over_seed(Image_name=image_input[0], path=UPLOAD_DIRECTORY, rmv_object_nuc=0.9,
                                               block_size=59, offset=norm_,
                                               block_size_cyto=9, offset_cyto=0.0004, global_ther=0.4,
                                               rmv_object_cyto=0.99, rmv_object_cyto_small=0.9, remove_border=True)
            nuc_s = AIPS_object.Nucleus_segmentation(ch2_, inv=False)
            offset_pred = norm_
            len_table = len(nuc_s['tabale_init'])
            if len_table > 3:
                break
        norm = np.random.normal(offset_pred, 0.001, 100)
        min_val = round(np.min(norm), 4)
        max_val = round(np.max(norm), 4)
        steps = (np.max(norm) - np.min(norm)) / bar_zoom_cyto
        value_marks = {i: i for i in [min_val, max_val]}
        return [min_val, max_val, value_marks, offset_pred,steps]
    else:
        if n_parm > 0:
            parameters = parse_contents(cont, pram)
            osc = parameters['offset_cyto'][0]
            norm = np.random.normal(osc, 0.001, 100)
            min_val = round(np.min(norm), 4)
            max_val = round(np.max(norm), 4)
            steps = (np.max(os) - np.min(os)) / bar_zoom_cyto
            value_marks = {i: i for i in [min_val, max_val]}
            offset_pred = os
        else:
            min_val = 0.001
            max_val = 0.8
            value_marks = {i: i for i in [0.001, 0.8]}
            offset_pred = offset_input
            steps = 0.001
        return [min_val, max_val, value_marks, offset_pred,steps]

if __name__ == "__main__":
    app.run_server()
    # app.run_server(debug=True)