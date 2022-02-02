'''
git add .
git commit -m "02-02-2021 main app - display image sync table sync no color"
##git push origin -u AIPS_dash_final
git push origin main
'''
from app import app
from app import server
from apps import Nuclues_count_predict, SVM_target_classification
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

from utils.Dash_functions import parse_contents
from utils.controls import controls, controls_nuc, controls_cyto, upload_parm
from utils import AIPS_functions as af
from utils import AIPS_module as ai
#import xml.etree.ElementTree as xml


UPLOAD_DIRECTORY = "app_uploaded_files"

next_display = dbc.Button('Next', id='display-val', n_clicks=0, color="Primary", className="me-1", style={'padding': '10px 24px'})
next_Nucleus = dbc.Button('Next', id='Nucleus-val', n_clicks=0, color="success", className="me-1", style={'padding': '10px 24px'})
next_Cell = dbc.Button('Next', id='Cell-val', n_clicks=0,color="Danger", className="me-1", style={'padding': '10px 24px'})
next_parameter = dbc.Button('Next', id='parameter-val', n_clicks=0, color="Info", className="me-1", style={'padding': '10px 24px'})
next_module = dbc.Button('Next', id='module-val', n_clicks=0, color="Secondary", className="me-1", style={'padding': '10px 24px'})


# loading jason
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


timeout = 10

app.layout = dbc.Container(
    [
        html.H1("Optical Pooled Cell Sorting Platform"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                 dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=True
                    ),
                html.Button('Submit', id='submit-val', n_clicks=0),
                dbc.Accordion(
                            [
                                dbc.AccordionItem(children=
                                [
                        controls
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
                dcc.Tabs(id = 'tabs', value = '',
                        children=[
                            #dcc.Tab(label="Image information",id = "Image-info-id", value="Image-info",style={'color': 'black'},selected_style={'color': 'red'},disabled=False),
                            dcc.Tab(label="Image display", id = "load_tab-id",value="load_tab",style={'color': 'black'},selected_style={'color': 'red'},disabled=False ),
                            dcc.Tab(label="Nucleus segmentation", id = "Nucleus-tab-id", value="Nucleus-tab",style={'color': 'black'},selected_style={'color': 'red'},disabled=True),
                            dcc.Tab(label="Cell segmentation", id = "Cell-tab-id", value="Cell-tab",style={'color': 'black'},selected_style={'color': 'red'},disabled=True),
                            dcc.Tab(label="Save image parameter", id = "save-id", value="save-tab",style={'color': 'black'},selected_style={'color': 'red'},disabled=True),
                            dcc.Tab(label="Select module for activation", id = "Module-tab-id", value="Module-tab",style={'color': 'black'},selected_style={'color': 'red'},disabled=False)
                    ]),
                html.Div(id='run_content'),
                dcc.Store(id='offset_store',data=None),
                dcc.Store(id='offset_cyto_store',data=None),
                dcc.Loading(html.Div(id='img-output'),type="circle",style={'height': '100%', 'width': '100%'}),
                html.Div(id="test-image-name",hidden=True),
                dcc.Interval(id = 'interval',interval=1000,max_intervals=2,disabled=True)
               # dcc.Loading(html.Img(id='img-output',style={'height': '100%', 'width': '100%'})),
            ])])])

# loading parameters file from local csv table
@app.callback(
    [Output('act_ch', 'value'),
    Output('block_size', 'value'),
    Output('offset_store', 'data'),
    Output('rmv_object_nuc', 'value'),
    Output('block_size_cyto', 'value'),
    Output('offset_cyto_store', 'data'),
    Output('global_ther', 'value'),
    Output('rmv_object_cyto', 'value'),
    Output('rmv_object_cyto_small', 'value')
     ],
    [Input('submit-parameters', 'n_clicks'),
     State('upload-csv', 'filename'),
     State('upload-csv', 'contents')])
def Load_image(n,pram,cont):
    if n is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,\
               dash.no_update,dash.no_update, dash.no_update, dash.no_update
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
    return channel,bs,os,ron,bsc,osc,gt,roc,rocs


@app.callback(
    Output('run_content', 'children'),
    [Input('submit-val', 'n_clicks'),
     State('upload-image', 'filename'),
     State('upload-image', 'contents')])
def Load_image(n,image,cont):
    if n is None:
        return dash.no_update
    for name, data_a in zip(image, cont):
        data = data_a.encode("utf8").split(b";base64,")[1]
        with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
            fp.write(base64.decodebytes(data))
    return [html.Button('Run', id='run-val', n_clicks=0)]

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
     State('Auto-nuc', 'on'),
     State('offset', 'value'),
     State('graduated-bar-slider', 'value'),
     State('upload-image', 'filename'),
     State('act_ch', 'value')])
def Updat_offset(set_n,bar_zoom,au,offset_input,bar_ind,image_input,channel_sel):
    if au:
        AIPS_object = ai.Segment_over_seed(Image_name=image_input[0], path=UPLOAD_DIRECTORY, rmv_object_nuc=0.9, block_size=59,
                                           offset=offset_input,block_size_cyto=9, offset_cyto=0.0004, global_ther=0.4, rmv_object_cyto=0.99,
                                           rmv_object_cyto_small=0.9, remove_border=True)
        img = AIPS_object.load_image()
        if channel_sel == 1:
            nuc_sel = '0'
            cyt_sel = '1'
        else:
            nuc_sel = '1'
            cyt_sel = '0'
        med_nuc = np.median(img[nuc_sel]) / 100
        norm = np.random.normal(med_nuc, 0.001*bar_ind, 100)
        offset_pred = []
        for norm_ in norm:
            AIPS_object = ai.Segment_over_seed(Image_name=image_input[0], path=UPLOAD_DIRECTORY, rmv_object_nuc=0.9,
                                               block_size=59, offset=norm_,
                                               block_size_cyto=9, offset_cyto=0.0004, global_ther=0.4,
                                               rmv_object_cyto=0.99, rmv_object_cyto_small=0.9, remove_border=True)
            img = AIPS_object.load_image()
            nuc_s = AIPS_object.Nucleus_segmentation(img['0'], inv=False)
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
     State('Auto-cyto', 'on'),
     State('offset_cyto', 'value'),
     State('graduated-bar-cyto', 'value'),
     State('upload-image', 'filename'),
     State('act_ch', 'value')])
def Updat_offset_cyto(set_n,bar_zoom_cyto,au,offset_input,bar_ind,image_input,channel_sel):
    if au:
        AIPS_object = ai.Segment_over_seed(Image_name=image_input[0], path=UPLOAD_DIRECTORY, rmv_object_nuc=0.9, block_size=59,
                                           offset=offset_input,block_size_cyto=9, offset_cyto=0.0004, global_ther=0.4, rmv_object_cyto=0.99,
                                           rmv_object_cyto_small=0.9, remove_border=True)
        img = AIPS_object.load_image()
        if channel_sel == 1:
            nuc_sel = '0'
            cyt_sel = '1'
        else:
            nuc_sel = '1'
            cyt_sel = '0'
        med_nuc = np.median(img[nuc_sel]) / 100
        norm = np.random.normal(med_nuc, 0.001*bar_ind, 100)
        offset_pred = []
        for norm_ in norm:
            AIPS_object = ai.Segment_over_seed(Image_name=image_input[0], path=UPLOAD_DIRECTORY, rmv_object_nuc=0.9,
                                               block_size=59, offset=norm_,
                                               block_size_cyto=9, offset_cyto=0.0004, global_ther=0.4,
                                               rmv_object_cyto=0.99, rmv_object_cyto_small=0.9, remove_border=True)
            img = AIPS_object.load_image()
            nuc_s = AIPS_object.Nucleus_segmentation(img['0'], inv=False)
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
        min_val = 0.001
        max_val = 0.8
        value_marks = {i: i for i in [0.001, 0.8]}
        offset_pred = offset_input
        steps = 0.001
        return [min_val, max_val, value_marks, offset_pred,steps]

@app.callback(
    Output('img-output', 'children'),
    [Input('run-val', 'n_clicks'),
    Input('tabs', 'value'),
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
     ])
def Parameters_initiation(nn,tab_input,image,cont,channel,int_on_nuc,high,low,bs,os,ron,bsc,int_on_cyto,osc,gt,roc,rocs):
    AIPS_object = ai.Segment_over_seed(Image_name=str(image[0]), path=UPLOAD_DIRECTORY, rmv_object_nuc=ron,
                                       block_size=bs,
                                       offset=os,
                                       block_size_cyto=bsc, offset_cyto=osc, global_ther=gt, rmv_object_cyto=roc,
                                       rmv_object_cyto_small=rocs, remove_border=False)
    img = AIPS_object.load_image()
    if channel==1:
        nuc_sel='0'
        cyt_sel='1'
    else:
        nuc_sel = '1'
        cyt_sel = '0'
    nuc_s = AIPS_object.Nucleus_segmentation(img[nuc_sel])
    seg = AIPS_object.Cytosol_segmentation(img[nuc_sel],img[cyt_sel],nuc_s['sort_mask'],nuc_s['sort_mask_bin'])
    # dict_ = {'img':img,'nuc':nuc_s,'seg':seg}
    # try to work on img
    ch_ = img[nuc_sel]
    ch2_ = img[cyt_sel]
    nmask2 = nuc_s['nmask2']
    nmask4 = nuc_s['nmask4']
    sort_mask = nuc_s['sort_mask']
    table = nuc_s['table']
    cell_mask_1 = seg['cell_mask_1']
    combine = seg['combine']
    cseg_mask = seg['cseg_mask']
    info_table = seg['info_table']
    mask_unfiltered = seg['mask_unfiltered']
    table_unfiltered = seg['table_unfiltered']
    image_size_x = np.shape(ch_)[1]
    image_size_y = np.shape(ch2_)[1]
    try:
        med_seed = int(np.median(seed_table))
        len_seed = len(seed_table)
    except:
        med_seed = 'None'
        len_seed = 'None'
    try:
        med_cyto = int(np.median(cyto_table))
        len_cyto = len(cyto_table)
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
    im_pil.save("1.png", format='png')  # this is for image processing
    encoded_image = base64.b64encode(open("1.png", 'rb').read())
    pix_2 = af.show_image_adjust(ch2_, low_prec=low, up_prec=high)
    pix_2 = pix_2 * 65535.000
    im_pil = Image.fromarray(np.uint16(pix_2))
    im_pil.save("2.png", format='png')  # this is for image processing
    encoded_image_ch2 = base64.b64encode(open("2.png", 'rb').read())
    '''
    Nucleus segmentation 
    '''
    encoded_image_nmask2 = af.save_pil_to_directory(nmask2, bit=1, mask_name='nmask2')
    encoded_image_sort_mask = af.save_pil_to_directory(sort_mask, bit=3, mask_name='sort_mask')
    '''
    Cytosol  segmentation 
    '''
    cell_mask_2 = np.where(cell_mask_1 == 1, True, False)
    combine = np.where(combine == 1, True, False)
    encoded_image_cell_mask_2 = af.save_pil_to_directory(cell_mask_2, bit=1, mask_name='_cell_mask')
    encoded_image_cell_combine = af.save_pil_to_directory(combine, bit=1, mask_name='_combine')
    encoded_image_cseg_mask = af.save_pil_to_directory(cseg_mask, bit=3, mask_name='_cseg')
    encoded_image_mask_unfiltered = af.save_pil_to_directory(mask_unfiltered, bit=3, mask_name='_csegg')
    len_unfiltered_table = table_unfiltered
    if tab_input == 'load_tab':
        return [
            dbc.Row([
                dbc.Col([
                    dbc.Col(
                    html.Label('Seed: ' + Channel_number_1, style={'text-align-last': 'center'})),
                    dbc.Col(
                    html.Img(id="img-load", src='data:image/png;base64,{}'.format(encoded_image.decode()),
                             style={'width': '90%', 'height': 'auto'}))]),
                dbc.Col([
                    dbc.Col(
                    html.Label('Target: ' + Channel_number_2, style={'text-align-last': 'center'})),
                    dbc.Col(
                    html.Img(id="img-load", src='data:image/png;base64,{}'.format(encoded_image_ch2.decode()),
                             style={'width': '90%', 'height': 'auto'}))]),
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
                dbc.Row(
                    dbc.Col(
                        next_Nucleus, width=3))
            ]
    elif tab_input == 'Nucleus-tab':
        return [
            dbc.Row([
                dbc.Col([
                    dbc.Col(html.Label('Seed Image', style={'text-align': 'right'})),
                    dbc.Col(html.Img(id="img-output-orig", src='data:image/png;base64,{}'.format(encoded_image.decode()),
                                     style={'width': '100%', 'height': 'auto'}, alt="re-adjust parameters", title="ORIG"))
                             ]),
                dbc.Col([
                    dbc.Col(html.Label('Local threshold map - seed', style={'text-align': 'right'})),
                    dbc.Col(html.Img(id="img-output-orig", src='data:image/png;base64,{}'.format(encoded_image_nmask2.decode()),
                                     style={'width': '100%', 'height': 'auto'}, alt="re-adjust parameters", title="ORIG"))
                                 ])
                    ]),
            dbc.Row([
                dbc.Col([
                dbc.Col(html.Label('RGB map - seed', style={'text-align': 'right'})),
                dbc.Col(
                    html.Img(id="img-output-orig", src='data:image/png;base64,{}'.format(encoded_image_sort_mask.decode()),
                             style={'width': '50%', 'height': 'auto'}, alt="re-adjust parameters", title="ORIG"))
                    ])
                    ]),
            dbc.Row(
                dbc.Col(
                    next_Cell, width=3))
                ]

    elif tab_input == 'Cell-tab':
        return [dbc.Row([

                        dbc.Col([html.Label('Target image', style={'text-align': 'center'}),
                        html.Br(),
                        html.Img(id="img-output-orig-target",
                                  src='data:image/png;base64,{}'.format(encoded_image_ch2.decode()),
                                  style={'width': '100%', 'height': 'auto'},
                                  alt="re-adjust parameters",
                                  title="Local Threshold")]),
                        dbc.Col([html.Label('Local threshold map - Target', style={'text-align': 'center'}),
                        html.Br(),
                        html.Img(id="img-output-mask-target",
                                  src='data:image/png;base64,{}'.format(encoded_image_cell_mask_2.decode()),
                                  style={'max-width': '100%', 'height': 'auto'},
                                  alt="re-adjust parameters",
                                  title='Local threshold map - Target')
                         ])

                                        ]),
                dbc.Row([
                    dbc.Col([html.Label('Mask - Target', style={'text-align': 'center'}),
                            html.Br(),
                            html.Img(id="img-output-mask-target",
                                          src='data:image/png;base64,{}'.format(encoded_image_mask_unfiltered.decode()),
                                          style={'max-width': '100%', 'height': 'auto'},
                                          alt="re-adjust parameters",
                                          title='Global Threshold - Target')
                                 ]),
                    dbc.Col([html.Label('Mask - Target (filterd)', style={'text-align': 'center'}),
                             html.Br(),
                             html.Img(id="img-output-mask-target",
                                      src='data:image/png;base64,{}'.format(encoded_image_cseg_mask.decode()),
                                      style={'max-width': '100%', 'height': 'auto'},
                                      alt="re-adjust parameters",
                                      title='Local threshold map - seed')
                             ])]),
            html.Br(),
            html.Br(),
            dbc.Row([
                dbc.Accordion([
                dbc.AccordionItem(
                    title="Cytosole segmentation inspection", children=[
                       dbc.Row(html.Label("Number of objects detected before filtering: {}".format(len_unfiltered_table.iloc[0,0]))),
                        dbc.Row(html.Label("Number of objects - large filtered: {}".format(len_unfiltered_table.iloc[0, 1]))),
                        dbc.Row(html.Label("Number of objects - Small filtered:: {}".format(len_unfiltered_table.iloc[0, 2])))
                                                                        ]
                                    )
                                    ])
                        ]),
            dbc.Row(
                dbc.Col(
                    next_parameter, width=3))
                ]
    elif tab_input == 'save-tab':
            with open('parameters.csv', 'w') as fp:
                pass
            return [
                    dbc.Row(html.P('Update the parameters.xml file from the parental folder:')),
                    html.Br(),
                    dbc.Row([
                        html.Button("Update", id="btn_update"),
                        html.Div(id='Progress',hidden=False)]),
                dbc.Row(
                    dbc.Col(
                        next_module, width=3))
                ]
    elif tab_input == "Module-tab":
        return [
            dbc.Accordion(
            [html.Div([
                dcc.Location(id='url', refresh=False),
                html.Div(id='page-content', children=[]),
                ]),
                dbc.AccordionItem(
                    title="Nucleus activation Module",
                    children=[
                            html.Div([
                                dcc.Link('Nucleus prediction test|', href='/apps/Nuclues_count_predict'),
                            ], className="row"),
                        daq.NumericInput(
                            id='Nuc_per_img',
                            label='Nucleus selected per image',
                            labelPosition='top',
                            value=10,
                        ),
                        daq.NumericInput(
                            id='Nuc_per_batch',
                            label='Total number of Nucleus',
                            labelPosition='top',
                            value=100,
                        ),
                        daq.BooleanSwitch(
                            on=True,
                            label="Save txt file",
                            labelPosition="top"
                        )
                    ]
                ),
                dbc.AccordionItem(
                    title="Target classification using SVM",
                    children=[
                        html.Div([
                            dcc.Location(id='url', refresh=False),
                            html.Div([
                                dcc.Link('Target classification|', href='/apps/SVM_target_classification'),
                            ], className="row"),
                            html.Div(id='page-content', children=[])
                        ])]
                ),
                dbc.AccordionItem(
                    "This is the content of the third section", title="Item 3"
                ),
            ],
            start_collapsed=True,
        )
        ]

@app.callback(
    Output('Progress', 'value'),
    [Input("btn_update", "n_clicks"),
    State('act_ch','value'),
    State('block_size','value'),
    State('offset','value'),
    State('rmv_object_nuc','value'),
    State('block_size_cyto', 'value'),
    State('offset_cyto', 'value'),
    State('global_ther', 'value'),
    State('rmv_object_cyto', 'value'),
    State('rmv_object_cyto_small', 'value'),
    ])
def Load_parameters_xml(nnn,channel,bs,os,ron,bsc,osc,gt,roc,rocs):
    dict = {'act_ch':[channel],'block_size':[bs],'offset':[os],'rmv_object_nuc':[ron],'block_size_cyto':[bsc],
            'offset_cyto':[osc], 'global_ther':[gt],
            'rmv_object_cyto':[roc],'rmv_object_cyto_small':[rocs]}
    #af.XML_creat('parameters.xml',str(bs),str(os),str(ron),str(bsc),str(osc),str(gt),str(roc),str(rocs))
    df = pd.DataFrame(dict)
    df.to_csv('parameters.csv', encoding='utf-8', index=False)
    return [html.P("done")]



#loading link
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/Nuclues_count_predict':
        return Nuclues_count_predict.layout
    elif pathname == '/apps/SVM_target_classification':
        return SVM_target_classification.layout
    else:
        return "No seed segment were detected, Readjust seed segmentation"



@app.callback(Output('load_tab-id', 'disabled'),
              Input('display-val', 'n_clicks'))
def activate_display_tab(next_display):
    if next_display < 1:
        disabled = True
    else:
        disabled = False
    return disabled

@app.callback(Output('Nucleus-tab-id', 'disabled'),
              Input('Nucleus-val', 'n_clicks'))
def activate_display_tab1(next_nuc):
    if next_nuc < 1:
        disabled = True
    else:
        disabled = False
    return disabled

@app.callback(Output('Cell-tab-id', 'disabled'),
              Input('Cell-val', 'n_clicks'))
def activate_display_tab2(next_cell):
    if next_cell < 1:
        disabled = True
    else:
        disabled = False
    return disabled

@app.callback(Output('save-id', 'disabled'),
              Input('parameter-val', 'n_clicks'))
def activate_display_tab3(next_save):
    if next_save < 1:
        disabled = True
    else:
        disabled = False
    return disabled

@app.callback(Output('Module-tab-id', 'disabled'),
              Input('module-val', 'n_clicks'))
def activate_display_tab4(next_mod):
    if next_mod < 1:
        disabled = True
    else:
        disabled = False
    return disabled

if __name__ == '__main__':
    app.run_server()