from dash_extensions.enrich import Dash, Output, Input, State, ServersideOutput
import time
import plotly.express as px
import json
import dash
from dash import ALL,MATCH
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
from utils.controls import controls, controls_nuc, controls_cyto, upload_parm, svm_slice_slider
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
                #svm_slice_slider,
                html.Div(id='image_data_holder', children=[]),
                html.Div(id='image_display_holder'),
                html.Div(id='tab_display'),
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
    [Output('run_content', 'children'),
    ServersideOutput('json_img_ch', 'data'),
    ServersideOutput('json_img_ch2', 'data')],
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
    json_object_img_ch = ch_
    json_object_img_ch2 = ch2_
    return [html.Button('Run', id='run-val', n_clicks=0)],json_object_img_ch,json_object_img_ch2



@app.callback(
    Output('image_data_holder', 'children'),
    [Input('json_img_ch', 'data'),
    Input('json_img_ch2', 'data'),
    State('image_data_holder', 'children')])
def image_chanked(ch,ch2,children):
    ch = np.array(ch)
    ch2 = np.array(ch2)
    H = np.shape(ch)[0]//2
    W = np.shape(ch)[1]//2
    tiles = [ch[x:x + H, y:y + W] for x in range(0, ch.shape[0], H) for y in range(0, ch.shape[1], W)]
    count = 0
    for tile in tiles:
        count += 1
        new_store = dcc.Store(id = {'type':'store_obj',
                                    'index':count},
                              data=tile)
        children.append(new_store)
    return children

@app.callback(
    Output('tab_display', 'children'),
    Input({'type': 'store_obj', 'index': ALL}, 'data')
)
def display_tab(data):
    count = np.linspace(1,len(data),len(data))
    return [html.Div(children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Slice number: {}".format(int(c)),
                                   id ={'type':'Image_number_slice',
                                        'index':int(c)}) for c in count])
                             ])
                        ])
                ]

@app.callback(
    Output('img-output', 'children'),
    Input({'type': 'store_obj', 'index': ALL}, 'data'),
    Input({'type': 'Image_number_slice', 'index': ALL}, 'title')
)
def display_output(data,title):
    ll = []
    count = 0
    for slice in data:
        count += 1
        temp_arr = np.array(slice)
        pix_2 = temp_arr * 65535.000
        im_pil = Image.fromarray(np.uint16(pix_2))
        fig_ch2 = px.imshow(im_pil, binary_string=True, binary_backend="jpg", width=250, height=250, title=str(count),
                            binary_compression_level=9).update_xaxes(showticklabels=False).update_yaxes(
                            showticklabels=False)
        ll.append(fig_ch2)
    return [html.Div(children=[
                dbc.Row([
                    dbc.Col(
                    dcc.Graph(id='slice_disp'+str(fig['layout']['title']['text']),figure=fig))
                    for fig in ll ])
                        ],
                    ) ]










# @app.callback(
#     Output('img-output', 'children'),
#     Input({'type': 'store_obj', 'index': ALL}, 'data')
# )
# def display_output(data):
#     ll = []
#     count = 0
#     for slice in data:
#         count += 1
#         temp_arr = np.array(slice)
#         pix_2 = temp_arr * 65535.000
#         im_pil = Image.fromarray(np.uint16(pix_2))
#         fig_ch2 = px.imshow(im_pil, binary_string=True, binary_backend="jpg", width=250, height=250, title=str(count),
#                             binary_compression_level=9).update_xaxes(showticklabels=False).update_yaxes(
#                             showticklabels=False)
#         ll.append(fig_ch2)
#     return [html.Div(children=[
#                 dbc.Row([
#                     dbc.Col(
#                     dcc.Graph(id='slice_disp'+str(fig['layout']['title']['text']),figure=fig))
#                     for fig in ll ])
#                         ],
#                     ) ]


#
# @app.callback(
#     Output('image_data_holder', 'children'),
#     [Input('json_img_ch', 'data'),
#     Input('json_img_ch2', 'data'),
#     State('image_data_holder', 'children')])
# def image_chanked(ch,ch2,children):
#     ch = np.array(ch)
#     ch2 = np.array(ch2)
#     H = np.shape(ch)[0]//2
#     W = np.shape(ch)[1]//2
#     tiles = [ch[x:x + H, y:y + W] for x in range(0, ch.shape[0], H) for y in range(0, ch.shape[1], W)]
#     new_store = dcc.Store(id = {'type':'store_obj',
#                                 'index':1},
#                           data=tiles[0])
#     children.append(new_store)
#     return children
#
#
#
#
#
# @app.callback(
#     Output('img-output', 'children'),
#     Input({'type': 'store_obj', 'index': ALL}, 'data')
# )
# def display_output(data):
#     ll = []
#     count = 0
#     for slice in data:
#         count += 1
#         temp_arr = np.array(slice)
#         pix_2 = temp_arr * 65535.000
#         im_pil = Image.fromarray(np.uint16(pix_2))
#         fig_ch2 = px.imshow(im_pil, binary_string=True, binary_backend="jpg", width=500, height=500, title=str(count),
#                             binary_compression_level=9).update_xaxes(showticklabels=False).update_yaxes(
#                             showticklabels=False)
#         ll.append(fig_ch2)
#     return [html.Div(children=[
#                 dbc.Col([dcc.Graph(id='slice_disp'+str(fig['layout']['title']['text']),figure=fig)])
#                         ],
#                     ) for fig in ll]



#
#
# @app.callback(
#     Output('img-output', 'children'),
#     [Input('run-val', 'n_clicks'),
#     Input('json_img_ch', 'data'),
#     Input('json_img_ch2', 'data'),
#     State('upload-image', 'filename'),
#     State('upload-image', 'contents'),
#     Input('act_ch', 'value'),
#     State('Auto-nuc', 'value'),
#     Input('high_pass', 'value'),
#     Input('low_pass', 'value'),
#     Input('block_size','value'),
#     Input('offset','value'),
#     Input('rmv_object_nuc','value'),
#     Input('block_size_cyto', 'value'),
#     State('Auto-cyto', 'value'),
#     Input('offset_cyto', 'value'),
#     Input('global_ther', 'value'),
#     Input('rmv_object_cyto', 'value'),
#     Input('rmv_object_cyto_small', 'value'),
#     Input('graduated-bar-slider-memory-scale','value'),
#      ])
# def Parameters_initiation(nn,ch,ch2, image,cont,channel,int_on_nuc,high,low,bs,os,ron,bsc,int_on_cyto,osc,gt,roc,rocs,memory_reduction):
#     ch_ = np.array(ch)
#     ch2_ = np.array(ch2)
#     pix = ch_ * 65535.000
#     im_pil = Image.fromarray(np.uint16(pix))
#     fig_ch = px.imshow(im_pil, binary_string=True, binary_backend="jpg",width=500,height=500,title='Seed:',binary_compression_level=9).update_xaxes(showticklabels = False).update_yaxes(showticklabels = False)
#     fig_ch.update_layout(title_x=0.5,dragmode="drawrect")
#     pix_2 = ch2_ * 65535.000
#     im_pil = Image.fromarray(np.uint16(pix_2))
#     fig_ch2 = px.imshow(im_pil, binary_string=True, binary_backend="jpg",width=500,height=500,title='Target:',binary_compression_level=9).update_xaxes(showticklabels = False).update_yaxes(showticklabels = False)
#     fig_ch2.update_layout(title_x=0.5,dragmode="drawrect")
#     return [
#             dbc.Row([
#                     dbc.Col(
#                         dcc.Graph(
#                             id="graph_ch",
#                             figure=fig_ch), md=6),
#                     dbc.Col(
#                         dcc.Graph(
#                             id="graph_ch2",
#                             figure=fig_ch2), md=6),
#                     ]),
#             ]

if __name__ == "__main__":
    app.run_server()

# if __name__ == "__main__":
#     app.run_server(debug=True)