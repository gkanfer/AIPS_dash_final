import dash_daq as daq
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


controls = dbc.Card(
        [
            html.Div(
                [
                dbc.Label("Image configuration"),
                    dcc.Checklist(
                        id='img_config',
                        options=[
                            {'label': 'Grayscale', 'value': 0},
                            {'label': 'RGB', 'value': 1},
                             ],
                        value=[0],
                        labelStyle={'display': 'inline-block'}
                    )
            ]),
            html.Div(
                [
                    dbc.Label("Choose seed channel"),
                    dcc.Checklist(
                        id='act_ch',
                        options=[
                            {'label': 'Channel 1', 'value': 1},
                            {'label': 'Channel 2', 'value': 2},
                        ],
                        value=[1],
                        labelStyle={'display': 'inline-block'}
                    )
                ]),
            html.Div(dbc.Label("Image intensity adjust")),
            html.Div(
                [
                    dbc.Label("high pass"),
                    dcc.Slider(
                        id='high_pass',
                        min=1,
                        max=99,
                        step=1,
                        marks={i: i for i in [20, 30, 40, 50 ,60 ,70 ,80]},
                        value=99
                        ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("low pass"),
                    dcc.Slider(
                        id='low_pass',
                        min=1,
                        max=99,
                        step=1,
                        marks={i: i for i in [20, 30, 40, 50 ,60 ,70 ,80]},
                        value=1,
                        ),
                ]
            )
        ], body=True)



controls_nuc = dbc.Card(
        [
         html.Div(
                [
                    dbc.Label("Nucleus segmentation"),
                    dbc.Label("Local Threshold:"),
                    dcc.Slider(
                        id='block_size',
                        min=1,
                        max=51,
                        step=2,
                        marks={i: i for i in [20, 30, 40, 50 ,60 ,70 ,80]},
                        value=13,
                        #tooltip = { 'always_visible': True }
                        ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Auto parameters initialise"),
                    dcc.Checklist(
                        id='Auto-nuc',
                        options=[
                            {'label': 'on', 'value': 1},
                            {'label': 'off', 'value': 0},
                        ],
                        value=[0],
                        labelStyle={'display': 'inline-block'}
                    )
            ]),
            html.Div(
                [
                    dbc.Label("Detect nuclei edges:"),
                    dcc.Slider(
                        id='offset',
                        min=0.000001,
                        max=0.9,
                        step=0.001,
                        marks={i: i for i in [0.01,0.8]},
                        value=0.001
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Remove small objects:"),
                    dcc.Slider(
                        id='rmv_object_nuc',
                        min=0.01,
                        max=0.99,
                        step=0.01,
                        marks={i: i for i in [0.01, 0.99]},
                        value=0.3
                    ),
                ]
            ),
         ],
        body=True,
    )


controls_cyto = dbc.Card(
        [
            html.Div(
                [
                    dbc.Label("Cytosol segmentation"),
                    dcc.Checklist(
                        id='cyto_seg',
                        options=[
                            {'label': 'on', 'value': 1},
                            {'label': 'off', 'value': 0}
                        ],
                        value=[1],
                        labelStyle={'display': 'inline-block'}
                    )
                ]),
            html.Div(
                [
                    dbc.Label("Auto parameters initialise"),
                    dcc.Checklist(
                        id='Auto-cyto',
                        options=[
                            {'label': 'on', 'value': 1},
                            {'label': 'off', 'value': 0},
                        ],
                        value=[1],
                        labelStyle={'display': 'inline-block'}
                    )
                ]),
            html.Div(
                [
                    dbc.Label("Local Threshold:"),
                    dcc.Slider(
                        id='block_size_cyto',
                        min=1,
                        max=51,
                        step=2,
                        marks={i: i for i in [20, 30, 40, 50, 60, 70, 80]},
                        value=13,
                        #tooltip = { 'always_visible': True }
                        ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Detect cytosol edges:"),
                    dcc.Slider(
                        id='offset_cyto',
                        min=0.000001,
                        max=0.9,
                        step=0.001,
                        marks={i: i for i in [0.001,0.9]},
                        value=0.001,
                        #tooltip = { 'always_visible': True }
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Detect global edges:"),
                    dcc.Slider(
                        id='global_ther',
                        min=0.01,
                        max=0.99,
                        step=0.01,
                        marks={i: i for i in [0.01, 0.99]},
                        value=0.3
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Remove large objects:"),
                    dcc.Slider(
                        id='rmv_object_cyto',
                        min=0.01,
                        max=0.99,
                        step=0.01,
                        marks={i: i for i in [0.01, 0.99]},
                        value=0.99
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Remove small objects:"),
                    dcc.Slider(
                        id='rmv_object_cyto_small',
                        min=0.01,
                        max=0.99,
                        step=0.01,
                        marks={i: i for i in [0.01, 0.99]},
                        value=0.99
                    ),
                ]
            ),

         ],
        body=True,
    )