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
from utils.controls import controls, controls_nuc, controls_cyto
from utils import AIPS_functions as af
from utils import AIPS_module as ai
#import xml.etree.ElementTree as xml

UPLOAD_DIRECTORY = "app_uploaded_files"

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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

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
                controls,
                controls_nuc,
                controls_cyto,
                ]),
            dbc.Col([
                dcc.Tabs(id = 'tabs', value = '',
                        children=[
                            dcc.Tab(label="Image information", value="Image-info",style={'color': 'black'},selected_style={'color': 'red'}, ),
                            dcc.Tab(label="Image display", value="load_tab",style={'color': 'black'},selected_style={'color': 'red'}, ),
                            dcc.Tab(label="Nucleus segmentation", value="Nucleus-tab",style={'color': 'black'},selected_style={'color': 'red'},),
                            dcc.Tab(label="Cell segmentation", value="Cell-tab",style={'color': 'black'},selected_style={'color': 'red'},),
                            dcc.Tab(label="Save image parameter", value="save-tab",style={'color': 'black'},selected_style={'color': 'red'},),
                            dcc.Tab(label="Select module for activation", value="Module-tab",style={'color': 'black'},selected_style={'color': 'red'},)
                    ]),
                html.Div(id='run_content'),
                dcc.Store(id='jason_ch'),
                dcc.Store(id='jason_ch2'),
                dcc.Store(id='jason_nmask2'),
                dcc.Store(id='jason_nmask4'),
                dcc.Store(id='jason_sort_mask'),
                dcc.Store(id='jason_table'),
                dcc.Store(id='jason_cell_mask_1'),
                dcc.Store(id='jason_combine'),
                dcc.Store(id='jason_cseg_mask'),
                dcc.Store(id='jason_info_table'),
                dcc.Loading(html.Div(id='img-output'),type="circle",style={'height': '100%', 'width': '100%'})
               # dcc.Loading(html.Img(id='img-output',style={'height': '100%', 'width': '100%'})),
            ])])])

@app.callback(
    Output('run_content', 'children'),
    [Input('submit-val', 'n_clicks'),
     State('upload-image', 'filename'),
     State('upload-image', 'contents')],
    prevent_intial_call=True)
def Load_image(n,image,cont):
    if image is None:
        raise dash.PreventUpdate
    else:
        for name, data_a in zip(image, cont):
            data = data_a.encode("utf8").split(b";base64,")[1]
            with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
                fp.write(base64.decodebytes(data))
        return [html.Button('Run', id='run-val', n_clicks=0)]

@app.callback(
    [Output('jason_ch', 'data'),
     Output('jason_ch2', 'data'),
     Output('jason_nmask2', 'data'),
     Output('jason_nmask4', 'data'),
     Output('jason_sort_mask', 'data'),
     Output('jason_table', 'data'),
     Output('jason_cell_mask_1', 'data'),
     Output('jason_combine', 'data'),
     Output('jason_cseg_mask', 'data'),
     Output('jason_info_table', 'data')],
    [Input('run-val', 'n_clicks'),
    State('upload-image', 'filename'),
    State('upload-image', 'contents'),
    Input('act_ch', 'value'),
    State('Auto-nuc', 'value'),
    Input('high_pass', 'value'),
    Input('low_pass', 'value'),
    Input('block_size','value'),
    Input('offset','value'),
    Input('rmv_object_nuc','value'),
    Input('cyto_seg', 'value'),
    Input('block_size_cyto', 'value'),
    State('Auto-cyto', 'value'),
    Input('offset_cyto', 'value'),
    Input('global_ther', 'value'),
    Input('rmv_object_cyto', 'value'),
    Input('rmv_object_cyto_small', 'value'),
     ], prevent_intial_call=False)
def Parameters_initiation(nn,image,cont,channel,int_on_nuc,high,low,bs,os,ron,cyt,bsc,int_on_cyto,osc,gt,roc,rocs):
    if nn is None:
        raise dash.PreventUpdate
    AIPS_object = ai.Segment_over_seed(Image_name=str(image[0]), path=UPLOAD_DIRECTORY, rmv_object_nuc=ron,
                                       block_size=bs,
                                       offset=os,
                                       block_size_cyto=bsc, offset_cyto=osc, global_ther=gt, rmv_object_cyto=roc,
                                       rmv_object_cyto_small=rocs, remove_border=False)
    img = AIPS_object.load_image()
    if channel[0]==1:
        nuc_sel='0'
        cyt_sel='1'
    else:
        nuc_sel = '1'
        cyt_sel = '0'
    nuc_s = AIPS_object.Nucleus_segmentation(img[nuc_sel])
    seg = AIPS_object.Cytosol_segmentation(img[nuc_sel],img[cyt_sel],nuc_s['sort_mask'],nuc_s['sort_mask_bin'])
    # dict_ = {'img':img,'nuc':nuc_s,'seg':seg}
    # try to work on img
    ch = img[nuc_sel].tolist()
    ch2 = img[cyt_sel].tolist()
    nmask2 = nuc_s['nmask2'].tolist()
    nmask4 = nuc_s['nmask4'].tolist()
    sort_mask = nuc_s['sort_mask'].tolist()
    table = nuc_s['table']
    cell_mask_1 = seg['cell_mask_1'].tolist()
    combine = seg['combine'].tolist()
    cseg_mask = seg['cseg_mask'].tolist()
    info_table = seg['info_table']
    json_object_ch = json.dumps(ch)
    json_object_ch2 = json.dumps(ch2)
    json_object_nmask2 = json.dumps(nmask2)
    json_object_nmask4 = json.dumps(nmask4)
    json_object_sort_mask = json.dumps(sort_mask)
    json_object_table = table.to_json(orient='split')
    json_object_cell_mask_1 = json.dumps(cell_mask_1)
    json_object_combine = json.dumps(combine)
    json_object_cseg_mask = json.dumps(cseg_mask)
    json_object_info_table = info_table.to_json(orient='split')
    return json_object_ch, json_object_ch2, json_object_nmask2, json_object_nmask4, json_object_sort_mask, \
           json_object_table, json_object_cell_mask_1, json_object_combine, json_object_cseg_mask, json_object_info_table

@app.callback(
    Output('img-output', 'children'),
    [Input('tabs', 'value'),
    Input('jason_ch', 'data'),
    Input('jason_ch2', 'data'),
    Input('jason_table', 'data'),
    Input('jason_info_table', 'data'),
    Input('jason_nmask2', 'data'),
    Input('jason_nmask4', 'data'),
    Input('jason_sort_mask', 'data'),
    Input('jason_cell_mask_1', 'data'),
    Input('jason_combine', 'data'),
    Input('jason_cseg_mask', 'data'),
    State('act_ch', 'value'),
    State('high_pass', 'value'),
    State('low_pass', 'value'),
     ], prevent_intial_call=True)
def Tab_content(tab_input, jason_data_ch, jason_data_ch2,jason_data_table,jason_data_info_table,
                jason_data_nmask2,jason_data_nmask4, jason_data_sort_mask, jason_data_cell_mask_1,
                jason_data_combine,jason_data_cseg_mask,channel,high,low):
    if tab_input is None:
        raise dash.ParseException
    elif tab_input == 'Image-info':
        seed_table = pd.read_json(jason_data_table,orient='split')['area']
        cyto_table = pd.read_json(jason_data_info_table,orient='split')['area']
        ch_ = json.loads(jason_data_ch)
        ch2_ = json.loads(jason_data_ch2)
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
        return [dbc.Col([
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
            ])])]
    elif tab_input == 'load_tab':
        ch_ = np.array(json.loads(jason_data_ch))
        ch2_ = np.array(json.loads(jason_data_ch2))
        if channel[0] == 1:
            Channel_number = 'Channel 1'
            x = ch_
        else:
            Channel_number = 'Channel 2'
            x = ch2_
        pix = af.show_image_adjust(x, low_prec=low, up_prec=high)
        pix = pix * 65535.000
        im_pil = Image.fromarray(np.uint16(pix))
        im_pil.save("1.png", format='png')  # this is for image processing
        encoded_image = base64.b64encode(open("1.png", 'rb').read())
        return [
            dbc.Row([
                html.Label(Channel_number, style={'text-align-last': 'center'}),
                html.Img(id="img-load", src='data:image/png;base64,{}'.format(encoded_image.decode()),
                         style={'width': '75%', 'height': 'auto'})
            ])
        ]
    elif tab_input == 'Nucleus-tab':
        nmask2 = np.array(json.loads(jason_data_nmask2))
        sort_mask = np.array(json.loads(jason_data_sort_mask))
        encoded_image_nmask2 = af.save_pil_to_directory(nmask2, bit=1,mask_name='nmask2')
        encoded_image_sort_mask = af.save_pil_to_directory(sort_mask, bit=3,mask_name='sort_mask')
        ch_ = np.array(json.loads(jason_data_ch)) * 65535.000
        ch2_ = np.array(json.loads(jason_data_ch2)) * 65535.000
        if channel[0] == 1:
            channel_mame = 'channel 1'
            encoded_image = af.save_pil_to_directory(ch_, bit=2,mask_name="orig")
        else:
            channel_mame = 'channel 2'
            encoded_image = af.save_pil_to_directory(ch2_, bit=2, mask_name="orig")
        return [dbc.Row([
                        dbc.Col([html.Label(channel_mame, style={'text-align': 'right'}),
                                html.Br(),
                                html.Img(id="img-output-orig",
                                      src='data:image/png;base64,{}'.format(encoded_image.decode()),
                                      style={'width': '100%','height':'auto'},
                                      alt="re-adjust parameters",
                                      title = "ORIG"),]),
                        dbc.Col([html.Label('Local threshold map - seed', style={'text-align': 'right'}),
                                html.Br(),
                                html.Img(id="img-output-nmask2",
                                        src='data:image/png;base64,{}'.format(encoded_image_nmask2.decode()),
                                        style={'max-width': '100%', 'height': 'auto'},
                                        alt="re-adjust parameters",
                                        title='Local threshold map - seed')
                                 ])
                                 ]),
                    dbc.Row([
                        dbc.Col([html.Label( 'RGB map - seed', style={'text-align': 'center'}),
                                html.Br(),
                                html.Img(id="img-output-sort_mask", style={'width': '100%','height':'auto'},
                                               src='data:image/png;base64,{}'.format(encoded_image_sort_mask.decode()),
                                                alt = "re-adjust parameters",
                                              title = 'RGB map - seed')]),
                        dbc.Col([html.Label('RGB map - seed', style={'text-align': 'center'},hidden=True),
                                 html.Br(),
                                 html.Img(id="img-output-sort_mask", style={'width': '100%', 'height': 'auto'},
                                          src='data:image/png;base64,{}'.format(encoded_image_sort_mask.decode()),
                                          alt="re-adjust parameters",
                                          title='RGB map - seed',
                                          hidden=True)])

                            ])
                        ]
    elif tab_input == 'Cell-tab':
        ch_ = np.array(json.loads(jason_data_ch)) * 65535.000
        ch2_ = np.array(json.loads(jason_data_ch2)) * 65535.000
        cell_mask_2 = np.array(json.loads(jason_data_cell_mask_1))
        cell_mask_2 = np.where(cell_mask_2 == 1, True, False)
        combine = np.array(json.loads(jason_data_combine))
        combine = np.where(combine == 1, True, False)
        cseg_mask = np.array(json.loads(jason_data_cseg_mask))
        encoded_image_cell_mask_2 = af.save_pil_to_directory(cell_mask_2, bit=1, mask_name='_cell_mask')
        encoded_image_cell_combine = af.save_pil_to_directory(combine, bit=1, mask_name='_combine')
        encoded_image_cseg_mask = af.save_pil_to_directory(cseg_mask, bit=3, mask_name='_cseg')
        if channel[0] == 1:
            Channel_number = 'Channel 1'
            x = ch_
        else:
            Channel_number = 'Channel 2'
            x = ch2_
        encoded_image = af.save_pil_to_directory(x, bit=2, mask_name="orig")
        return [dbc.Row([
                        dbc.Col([html.Label(Channel_number, style={'text-align': 'center'}),
                        html.Br(),
                        html.Img(id="img-output-orig-target",
                                  src='data:image/png;base64,{}'.format(encoded_image.decode()),
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
                         ])]),
                dbc.Row([
                    dbc.Col([html.Label('Global Threshold - Target', style={'text-align': 'center'}),
                            html.Br(),
                            html.Img(id="img-output-mask-target",
                                          src='data:image/png;base64,{}'.format(encoded_image_cell_combine.decode()),
                                          style={'max-width': '100%', 'height': 'auto'},
                                          alt="re-adjust parameters",
                                          title='Global Threshold - Target')
                                 ]),
                    dbc.Col([html.Label('Mask - Target', style={'text-align': 'center'}),
                             html.Br(),
                             html.Img(id="img-output-mask-target",
                                      src='data:image/png;base64,{}'.format(encoded_image_cseg_mask.decode()),
                                      style={'max-width': '100%', 'height': 'auto'},
                                      alt="re-adjust parameters",
                                      title='Local threshold map - seed')
                             ])])

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
                    ]
    elif tab_input == "Module-tab":
        return [
            dbc.Accordion(
            [
                dbc.AccordionItem(
                    title="Nucleus activation Module",
                    children=[
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
                    "This is the content of the second section", title="Item 2"
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
    State('block_size','value'),
    State('offset','value'),
    State('rmv_object_nuc','value'),
    State('block_size_cyto', 'value'),
    State('offset_cyto', 'value'),
    State('global_ther', 'value'),
    State('rmv_object_cyto', 'value'),
    State('rmv_object_cyto_small', 'value'),
    ], prevent_intial_call=False)
def Load_parameters_xml(nnn,bs,os,ron,bsc,osc,gt,roc,rocs):
    dict = {'block_size':[bs],'offset':[os],'rmv_object_nuc':[ron],'block_size_cyto':[bsc],
            'offset_cyto':[osc], 'global_ther':[gt],
            'rmv_object_cyto':[roc],'rmv_object_cyto_small':[rocs]}
    #af.XML_creat('parameters.xml',str(bs),str(os),str(ron),str(bsc),str(osc),str(gt),str(roc),str(rocs))
    df = pd.DataFrame(dict)
    df.to_csv('parameters.csv', encoding='utf-8', index=False)
    return [html.P("done")]
# 01/17/22 c

if __name__ == '__main__':
    app.run_server()