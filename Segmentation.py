import skimage.transform
import tifffile as tfi
import numpy as np
from skimage.filters import threshold_local
from scipy.ndimage.morphology import binary_opening
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import measure
import os
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
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
from scipy.ndimage.morphology import binary_opening, binary_erosion, binary_dilation
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
from utils.Display_composit import image_with_contour, countor_map

path = '/Users/kanferg/Desktop/NIH_Youle/Python_projacts_general/dash/AIPS_Dash_Final/app_uploaded_files/'
#Composite.tif10.tif
AIPS_object = ai.Segment_over_seed(Image_name='dmsot0273_0003-512.tif', path=path, rmv_object_nuc=0.5, block_size=83,
                                           offset=0.00001,block_size_cyto=11, offset_cyto=-0.0003, global_ther=0.4, rmv_object_cyto=0.99,
                                           rmv_object_cyto_small=0.9, remove_border=False)
img = AIPS_object.load_image()
nuc_s = AIPS_object.Nucleus_segmentation(img['1'], inv=False)
ch = img['1']
ch2 = img['0']

nuc_s = AIPS_object.Nucleus_segmentation(img['1'], inv=False, for_dash=False,rescale_image=True )
seg = AIPS_object.Cytosol_segmentation(ch, ch2, nuc_s['sort_mask'], nuc_s['sort_mask_bin'], rescale_image=True)
# dict_ = {'img':img,'nuc':nuc_s,'seg':seg}
# try to work on img
nmask2 = nuc_s['nmask2']
nmask4 = nuc_s['nmask4']
sort_mask_bin = nuc_s['sort_mask_bin']
sort_mask = nuc_s['sort_mask']
table = nuc_s['table']
cell_mask_1 = seg['cell_mask_1']
combine = seg['combine']
cseg_mask = seg['cseg_mask']
cseg_mask_bin = seg['cseg_mask_bin']
info_table = seg['info_table']
mask_unimport skimage.transform
import tifffile as tfi
import numpy as np
from skimage.filters import threshold_local
from scipy.ndimage.morphology import binary_opening
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import measure
import os
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
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
from scipy.ndimage.morphology import binary_opening, binary_erosion, binary_dilation
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
from utils.Display_composit import image_with_contour, countor_map

path = '/Users/kanferg/Desktop/NIH_Youle/Python_projacts_general/dash/AIPS_Dash_Final/app_uploaded_files/'
#Composite.tif10.tif
AIPS_object = ai.Segment_over_seed(Image_name='dmsot0273_0003-512.tif', path=path, rmv_object_nuc=0.5, block_size=83,
                                           offset=0.00001,block_size_cyto=11, offset_cyto=-0.0003, global_ther=0.4, rmv_object_cyto=0.99,
                                           rmv_object_cyto_small=0.9, remove_border=False)
img = AIPS_object.load_image()
nuc_s = AIPS_object.Nucleus_segmentation(img['1'], inv=False)
ch = img['1']
ch2 = img['0']

nuc_s = AIPS_object.Nucleus_segmentation(img['1'], inv=False, for_dash=False,rescale_image=True )
seg = AIPS_object.Cytosol_segmentation(ch, ch2, nuc_s['sort_mask'], nuc_s['sort_mask_bin'], rescale_image=True)
# dict_ = {'img':img,'nuc':nuc_s,'seg':seg}
# try to work on img
nmask2 = nuc_s['nmask2']
nmask4 = nuc_s['nmask4']
sort_mask_bin = nuc_s['sort_mask_bin']
sort_mask = nuc_s['sort_mask']
table = nuc_s['table']
cell_mask_1 = seg['cell_mask_1']
combine = seg['combine']
cseg_mask = seg['cseg_mask']
cseg_mask_bin = seg['cseg_mask_bin']
info_table = seg['info_table']
mask_unfiltered = seg['mask_unfiltered']
filtered = seg['mask_unfiltered']
table_unfiltered = seg['table_unfiltered']

#
# fig, ax = plt.subplots(2, 3, figsize=(12, 12))
# ax[0][0].imshow(nuc_s['nmask2'], cmap=plt.cm.gray)
# ax[0][1].imshow(nuc_s['nmask4'], cmap=plt.cm.gray)
# ax[0][2].imshow(nuc_s['sort_mask'], cmap=plt.cm.gray)
#
# ax[1][0].imshow(seg['cell_mask_1'], cmap=plt.cm.gray)
# ax[1][1].imshow(seg['cseg_mask'], cmap=plt.cm.gray)
# ax[1][2].imshow(seg['cseg_mask_bin'], cmap=plt.cm.gray)


im_pil_nmask2 = af.px_pil_figure(nmask2, bit=1, mask_name='nmask2', fig_title='Local threshold map - seed',
                                        wh=500)
#im_pil_nmask2.show()
#plt.imshow(sort_mask_bin)


sort_mask_bin = dx.outline_seg(sort_mask_bin,binary_img=True)
ch1_sort_mask = af.rgb_file_gray_scale(ch, mask=sort_mask_bin, channel=0,bin_composite=True)
fig_im_pil_sort_mask = af.px_pil_figure(ch1_sort_mask, bit=3, mask_name='sort_mask', fig_title='RGB map - seed',wh=500)
fig_im_pil_sort_mask.show()



cseg_mask_bin = dx.outline_seg(cseg_mask_bin, binary_img=True)
ch2_cseg_mask = af.rgb_file_gray_scale(ch2, mask=cseg_mask_bin, channel=0,bin_composite=True)
fig_im_pil_cseg_mask = af.px_pil_figure(ch2_cseg_mask, bit=3, mask_name='_cseg',
                                                fig_title='Mask - Target (filterd)', wh=500)
fig_im_pil_cseg_mask.show()


ch2_mask_unfiltered = af.rgb_file_gray_scale(ch2, mask=mask_unfiltered, channel=0,bin_composite=False)
fig_im_pil_mask_unfiltered = af.px_pil_figure(ch2_mask_unfiltered, bit=3, mask_name='_csegg', fig_title='Mask - Target',
                                              wh=500)
fig_im_pil_mask_unfiltered.show()

# med_nuc = np.median(ch) / 100
# norm = np.random.normal(med_nuc, 0.001*10, 100)
# offset_pred = []
# count = 0
# for norm_ in norm:
#     count += 1
#     AIPS_object = ai.Segment_over_seed(Image_name='dmsot0273_0003-512.tif', path=path, rmv_object_nuc=0.9,
#                                        block_size=59, offset=norm_,
#                                        block_size_cyto=9, offset_cyto=0.0004, global_ther=0.4,
#                                        rmv_object_cyto=0.99, rmv_object_cyto_small=0.9, remove_border=True)
#     nuc_s = AIPS_object.Nucleus_segmentation(ch, inv=False)
#     offset_pred = norm_
#     len_table = len(nuc_s['tabale_init'])
#     if len_table==3:
#         break
#
#
#
#
#
#
#
#
#
#
#
#
# # fig, ax = plt.subplots(2, 3, figsize=(12, 12))
# # ax[0][0].imshow(nuc_s['nmask2'], cmap=plt.cm.gray)
# # ax[0][1].imshow(nuc_s['nmask4'], cmap=plt.cm.gray)
# # ax[0][2].imshow(nuc_s['sort_mask'], cmap=plt.cm.gray)
# y = nuc_s['sort_mask']
# plt.imshow(y)
# x = skimage.transform.resize(y, (y.shape[0] * 4, y.shape[1] *4),anti_aliasing=False,preserve_range=True)
# x = np.asarray(x,np.int32)
# plt.imshow(x)
# y = nuc_s['sort_mask']
#
# ch2 = (ch2 / ch2.max()) * 255
# ch2_u8 = np.uint8(ch2)
# rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
# rgb_input_img[:, :, 0] = ch2_u8
# rgb_input_img[:, :, 1] = ch2_u8
# rgb_input_img[:, :, 2] = ch2_u8
# bf_mask = dx.binary_frame_mask(ch2_u8, x)
# bf_mask = np.where(bf_mask == 1, True, False)
# rgb_input_img[bf_mask > 0, 2] = 255
#
#
# sort_mask_bin=nuc_s['sort_mask_bin']
# seg_mask_temp = np.zeros(np.shape(sort_mask_bin), dtype=np.int64)
# seg_mask_eros_9 = binary_erosion(sort_mask_bin, structure=np.ones((9, 9))).astype(np.float64)
# seg_mask_eros_3 = binary_erosion(sort_mask_bin, structure=np.ones((3, 3))).astype(np.float64)
# framed_mask = seg_mask_eros_3 - seg_mask_eros_9
# plt.imshow(framed_mask)
# x = skimage.transform.resize(framed_mask, (framed_mask.shape[0] * 4, framed_mask.shape[1] *4),anti_aliasing=False,preserve_range=True)
# plt.imshow(x)
# bf_mask = np.where(bf_mask == 1, True, False)
#
# ch2 = (ch2 / ch2.max()) * 255
# ch2_u8 = np.uint8(ch2)
# rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
# rgb_input_img[:, :, 0] = ch2_u8
# rgb_input_img[:, :, 1] = ch2_u8
# rgb_input_img[:, :, 2] = ch2_u8
# bf_mask = np.where(x > 0, True, False)
# rgb_input_img[bf_mask > 0, 2] = 255
# plt.imshow(rgb_input_img)
#
#
# def test_x(x):
#     if x==0:
#         x='if'
#     elif x==1:
#         x='elif'
#     return x
# print(test_x(76))
#
#
#
#
#
#
#
#
#
#
#
# #
# # fig, ax = plt.subplots(2, 3, figsize=(12, 12))
# # ax[0][0].imshow(seg['cell_mask_1'], cmap=plt.cm.gray)
# # ax[0][1].imshow(seg['cseg_mask'], cmap=plt.cm.gray)
# # ax[0][2].imshow(seg['mask_unfiltered'], cmap=plt.cm.gray)
#


#
# input_gs_image = (ch / ch.max()) * 255
# ch2_u8 = np.uint8(input_gs_image)
# import skimage
# from skimage.transform import rescale, resize, downscale_local_mean
# image_rescaled = skimage.transform.rescale(ch, 0.0625, anti_aliasing=False)
# %timeit x  = threshold_local(image_rescaled, 13, "gaussian", 0.01)


