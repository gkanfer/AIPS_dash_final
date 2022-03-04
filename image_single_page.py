from dash_extensions.enrich import Dash, Output, Input, State, ServersideOutput
import time
import plotly.express as px
import json
import dash
import dash_bootstrap_components as dbc
import tifffile as tfi
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from PIL import Image
import plotly.express as px
import pathlib
import base64
import pandas as pd
import io
from io import BytesIO
from utils.controls import controls, controls_nuc, controls_cyto, upload_parm
from utils.Dash_functions import parse_contents
from utils import AIPS_functions as af
from utils import AIPS_module as ai

from dash import html, dcc



app = dash.Dash(prevent_initial_callbacks=True,
           external_stylesheets=[dbc.themes.JOURNAL, dbc.icons.FONT_AWESOME],)

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
                html.Div([
                    dbc.Alert(id='alert_display', is_open=False),
                    dcc.Loading(html.Div(id='img-output'), type="circle", style={'height': '100%', 'width': '100%'}),
                ]),
                html.Div(id='run_content'),
                dcc.Store(id='json_img_ch',data=None),
                dcc.Store(id='json_img_ch2',data=None),
                dcc.Store(id='json_react', data=None), # rectangle for memory reduction
                dcc.Store(id='offset_store',data=None),
                dcc.Store(id='offset_cyto_store',data=None),
                dcc.Store(id='slider-memory-scale', data=None),
                html.Div(id="test-image-name",hidden=True),
                dcc.Interval(id = 'interval',interval=1000,max_intervals=2,disabled=True)
            ]),
            ])], fluid=True)

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
    Output('slider-memory-scale','data'),
     ],
    [Input('submit-parameters', 'n_clicks'),
     State('upload-csv', 'filename'),
     State('upload-csv', 'contents')])
def Load_image(n,pram,cont):
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
    mem = parameters['memory_reduction'][0] #(1,2,3,4)
    set_nuc=1
    set_cyt=1
    return channel,bs,os,ron,bsc,osc,gt,roc,rocs,set_nuc,set_cyt,mem


@app.callback(
    [Output('run_content', 'children'),
    Output('json_img_ch', 'data'),
    Output('json_img_ch2', 'data')],
    [Input('submit-val', 'n_clicks'),
     State('upload-image', 'filename'),
     State('upload-image', 'contents'),
     Input('act_ch', 'value'),
     Input('json_react','data'),
     ])
def Load_image(n,image,cont,channel_sel,react):
    '''
    react: reactangle from draw compnante of user
    '''
    if n == 0:
        return dash.no_update,dash.no_update,dash.no_update
    content_string = cont[0].split('data:image/tiff;base64,')[1]
    decoded = base64.b64decode(content_string)
    pixels = tfi.imread(io.BytesIO(decoded))
    pixels_float = pixels.astype('float64')
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
    json_object_img_ch = json.dumps(ch_.tolist())
    json_object_img_ch2 = json.dumps(ch2_.tolist())
    return [html.Button('Run', id='run-val', n_clicks=0)],json_object_img_ch,json_object_img_ch2

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
     State('graduated-bar-slider-memory-scale', 'value'),
     ])
def Updat_offset(set_n,bar_zoom,ch,au,offset_input,bar_ind,image_input,channel_sel,n_parm,pram,cont,memory_reduction):
    if au:
        memory_index = {1: [0.25, 4], 2: [0.125, 8], 3: [0.062516, 16], 4: [0.031258, 32]}
        ch_ = np.array(json.loads(ch))
        med_nuc = np.median(ch_) / 400
        norm = np.random.normal(med_nuc, 0.001*bar_ind, 100)
        offset_pred = []
        for norm_ in norm:
            AIPS_object = ai.Segment_over_seed(Image_name=image_input[0], path=UPLOAD_DIRECTORY, rmv_object_nuc=0.9,
                                               block_size=59, offset=norm_,
                                               block_size_cyto=9, offset_cyto=0.0004, global_ther=0.4,
                                               rmv_object_cyto=0.99, rmv_object_cyto_small=0.9, remove_border=True)
            nuc_s = AIPS_object.Nucleus_segmentation(ch_, inv=False,rescale_image=True,scale_factor=memory_index[memory_reduction])
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
    State('graduated-bar-slider-memory-scale', 'value'),
     ])
def Updat_offset_cyto(set_n,bar_zoom_cyto,ch2,au,offset_input,bar_ind,image_input,channel_sel,n_parm,pram,cont,memory_reduction):
    if au:
        memory_index = {1: [0.25, 4], 2: [0.125, 8], 3: [0.062516, 16], 4: [0.031258, 32]}
        ch2_ = np.array(json.loads(ch2))
        med_nuc = np.median(ch2_) / 400
        norm = np.random.normal(med_nuc, 0.001*bar_ind, 100)
        offset_pred = []
        for norm_ in norm:
            AIPS_object = ai.Segment_over_seed(Image_name=image_input[0], path=UPLOAD_DIRECTORY, rmv_object_nuc=0.9,
                                               block_size=59, offset=norm_,
                                               block_size_cyto=9, offset_cyto=0.0004, global_ther=0.4,
                                               rmv_object_cyto=0.99, rmv_object_cyto_small=0.9, remove_border=True)
            nuc_s = AIPS_object.Nucleus_segmentation(ch2_, inv=False,rescale_image=True,scale_factor=memory_index[memory_reduction])
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

@app.callback(
    [Output('img-output', 'children'),
     Output('alert_display', 'is_open'),
     Output('nuc', 'active'),
     Output('nuc', 'disabled'),],
    [Input('run-val', 'n_clicks'),
    Input('json_img_ch', 'data'),
    Input('json_img_ch2', 'data'),
    State('upload-image', 'filename'),
    State('upload-image', 'contents'),
    Input('act_ch', 'value'),
    State('Auto-nuc', 'value'),
    Input('high_pass', 'value'),
    Input('low_pass', 'value'),
    Input('block_size','value'),
    Input('offset','value'),
    Input('rmv_object_nuc','value'),
    Input('block_size_cyto', 'value'),
    State('Auto-cyto', 'value'),
    Input('offset_cyto', 'value'),
    Input('global_ther', 'value'),
    Input('rmv_object_cyto', 'value'),
    Input('rmv_object_cyto_small', 'value'),
    Input('graduated-bar-slider-memory-scale','value'),
     ])
def Parameters_initiation(nn,ch,ch2, image,cont,channel,int_on_nuc,high,low,bs,os,ron,bsc,int_on_cyto,osc,gt,roc,rocs,memory_reduction):
    memory_index =  {1:[0.25,4],2:[0.125,8],3:[0.062516,16],4:[0.031258,32]}
    AIPS_object = ai.Segment_over_seed(Image_name=image[0], path=UPLOAD_DIRECTORY,rmv_object_nuc=ron,
                                       block_size=bs,
                                       offset=os,
                                       block_size_cyto=bsc, offset_cyto=osc, global_ther=gt, rmv_object_cyto=roc,
                                       rmv_object_cyto_small=rocs, remove_border=False)
    ch_ = np.array(json.loads(ch))
    ch2_ = np.array(json.loads(ch2))
    if np.shape(ch_)[0] > 512:
        alert_massage = True
    else:
        alert_massage = False
    ch_3c = af.gray_scale_3ch(ch_)
    ch2_3c = af.gray_scale_3ch(ch2_)
    nuc_s = AIPS_object.Nucleus_segmentation(ch_,rescale_image=True,scale_factor=memory_index[memory_reduction])
    #seg = AIPS_object.Cytosol_segmentation(ch_, ch2_, nuc_s['sort_mask'], nuc_s['sort_mask_bin'], rescale_image=True)
    # dict_ = {'img':img,'nuc':nuc_s,'seg':seg}
    # try to work on img
    nmask2 = nuc_s['nmask2']
    nmask4 = nuc_s['nmask4']
    sort_mask = nuc_s['sort_mask']
    sort_mask_bin = nuc_s['sort_mask_bin']
    sort_mask_bin = np.array(sort_mask_bin, dtype=np.int8)
    table = nuc_s['table']
    seg = AIPS_object.Cytosol_segmentation(ch_, ch2_, sort_mask, sort_mask_bin, rescale_image=True,scale_factor=memory_index[memory_reduction])
    cell_mask_1 = seg['cell_mask_1']
    combine = seg['combine']
    cseg_mask = seg['cseg_mask']
    info_table = seg['info_table']
    mask_unfiltered = seg['mask_unfiltered']
    table_unfiltered = seg['table_unfiltered']
    image_size_x = np.shape(ch_)[1]
    image_size_y = np.shape(ch2_)[1]
    try:
        med_seed = int(np.median(table['area']))
        len_seed = len(table)
    except:
        med_seed = 'None'
        len_seed = 'None'
    try:
        med_cyto = int(np.median(info_table['area']))
        len_cyto = len(info_table)
    except:
        med_cyto = 'None'
        len_cyto = 'None'
    '''
    Display image
    '''
    if channel == 1:
        Channel_number_1 = 'Channel 1'
        Channel_number_2 = 'Channel 2'
    else:
        Channel_number_1 = 'Channel 2'
        Channel_number_2 = 'Channel 1'
    pix = af.show_image_adjust(ch_, low_prec=low, up_prec=high)
    pix = pix * 65535.000
    im_pil = Image.fromarray(np.uint16(pix))
    fig_ch = px.imshow(im_pil, binary_string=True, binary_backend="jpg",width=500,height=500,title='Seed:'+ Channel_number_1,binary_compression_level=9).update_xaxes(showticklabels = False).update_yaxes(showticklabels = False)
    fig_ch.update_layout(title_x=0.5,dragmode="drawrect")
    pix_2 = af.show_image_adjust(ch2_, low_prec=low, up_prec=high)
    pix_2 = pix_2 * 65535.000
    im_pil = Image.fromarray(np.uint16(pix_2))
    fig_ch2 = px.imshow(im_pil, binary_string=True, binary_backend="jpg",width=500,height=500,title='Target:'+ Channel_number_2,binary_compression_level=9).update_xaxes(showticklabels = False).update_yaxes(showticklabels = False)
    fig_ch2.update_layout(title_x=0.5,dragmode="drawrect")
    return [
            dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            id="graph_ch",
                            figure=fig_ch), md=6),
                    dbc.Col(
                        dcc.Graph(
                            id="graph_ch2",
                            figure=fig_ch2), md=6),
                    ]),
            html.Br(),
            html.Br(),
            dbc.Col([
                dbc.Row(html.Label('Image parameters:')),
            html.P([
                dbc.Row(html.Label("Image size: {} x {}".format(image_size_x, image_size_y))),
                html.Br(),
                dbc.Row(html.Label('Seed Image parameters:')),
                html.P([
                    dbc.Row(html.Label("Median object area: {}".format(med_seed))),
                    dbc.Row(html.Label("Number of objects detected: {}".format(len_seed))), ]),
                dbc.Row(html.Label('Target Image Parameters :')),
                html.P([
                    dbc.Row(html.Label("Median object area: {}".format(med_cyto))),
                    dbc.Row(html.Label("Number of objects detected: {}".format(len_cyto)))]),
            ])]),
               # daq.BooleanSwitch(
               #     label='Clear selections',
               #     id='clear_selc',
               #     disabled=False,
               #     on=False,
               # )
            ],alert_massage,True,False

if __name__ == "__main__":
    app.run_server()

# if __name__ == "__main__":
#     app.run_server(debug=True)