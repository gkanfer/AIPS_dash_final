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



# Set up the app
external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/object_properties_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# make image and Mask
path = '/Users/kanferg/Desktop/NIH_Youle/Python_projacts_general/dash/AIPS_Dash_Final/app_uploaded_files/'
#Composite.tif10.tif
AIPS_object = ai.Segment_over_seed(Image_name='dmsot0273_0003-512.tif', path=path, rmv_object_nuc=0.5, block_size=59,
                                           offset=-0.0004,block_size_cyto=59, offset_cyto=-0.0004, global_ther=0.2, rmv_object_cyto=0.1,
                                           rmv_object_cyto_small=0.1, remove_border=False)

img = AIPS_object.load_image()
nuc_s = AIPS_object.Nucleus_segmentation(img['1'], inv=False)

ch = img['1']
ch2 = img['0']
#ch2 = ch2*2**16
ch2 = (ch2/ch2.max())*255
ch2 = np.uint8(ch2)
composite = np.zeros((np.shape(ch2)[0],np.shape(ch2)[1],3),dtype=np.uint8)
composite[:,:,0] = ch2
composite[:,:,1] = ch2
composite[:,:,2] = ch2
img = composite
# mask
label_array = nuc_s['sort_mask']
current_labels = np.unique(label_array)[np.nonzero(np.unique(label_array))]
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
prop_table = measure.regionprops_table(
    label_array, intensity_image=img, properties=prop_names
)
table = pd.DataFrame(prop_table)
columns = [
    {"name": label_name, "id": label_name,"type": "numeric", "selectable": True}
    for label_name in table.columns]
df = table

# Format the Table columns
# columns = [
#     {"name": label_name, "id": label_name, "selectable": True}
#     if precision is None
#     else {
#         "name": label_name,
#         "id": label_name,
#         "type": "numeric",
#         "selectable": True,
#     }
#     for label_name, precision in zip(prop_names, (None, None, 4, 4, None, 3))]
columns = [{"name": i, "id": i} for i in table.columns]
initial_columns = ["label", "area"]
x=[1]
z = [
    {
         'if': {'filter_query': '{{label}} = {}'.format(i)},
            'backgroundColor': '#85144b',
            'color': 'white'
    } for i in x
    ]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = dash_table.DataTable(
                                    id="table-line",
                                    columns=columns,
                                    data=table.to_dict("records"),
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
                                    # selected_columns=initial_columns,df.loc[df['label']==1,['label']]
                                    # style_data_conditional=[
                                    #         {
                                    #              'if': {'filter_query': '{{label}} = {}'.format(i)},
                                    #                 'backgroundColor': '#85144b',
                                    #                 'color': 'white'
                                    #         } for i in x
                                    #     ],
                                    style_data_conditional = z,
                                    style_table={"overflowY": "scroll"},
                                    fixed_rows={"headers": False, "data": 0},
                                    style_cell={"width": "85px"},
                                    page_size=10,
                                )

if __name__ == '__main__':
    app.run_server()



from dash import Dash, html, Input, Output, callback_context

app = Dash(__name__)
