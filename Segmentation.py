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


def plot_composite_image(img,mask,fig_title,alpha=0.2):
    # apply colors to mask
    mask = np.array(mask, dtype=np.int32)
    mask_deci = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    cm = plt.get_cmap('CMRmap')
    colored_image = cm(mask_deci)
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
    #RGB pil image
    img_mask = img_as_ubyte(colored_image)
    im_mask_pil = Image.fromarray(img_mask).convert('RGB')
    img_gs = img_as_ubyte(img)
    im_pil = Image.fromarray(img_gs).convert('RGB')
    im3 = Image.blend(im_pil, im_mask_pil, alpha)
    fig_ch = px.imshow(im3, binary_string=True, binary_backend="jpg", width=500, height=500, title=fig_title,
                       binary_compression_level=9).update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig_ch.update_layout(title_x=0.5)
    return fig_ch




path = '/Users/kanferg/Desktop/NIH_Youle/Python_projacts_general/dash/AIPS_Dash_Final/app_uploaded_files/'
#Composite.tif10.tif
AIPS_object = ai.Segment_over_seed(Image_name='dmsot0273_0003-512.tif', path=path, rmv_object_nuc=0.5, block_size=83,
                                           offset=0.00001,block_size_cyto=17, offset_cyto=-0.004, global_ther=0.4, rmv_object_cyto=0.7,
                                           rmv_object_cyto_small=0.1, remove_border=True)
img = AIPS_object.load_image()
ch = img['1']
ch2 = img['0']
H = ch.shape[0]//2
W = ch.shape[1]//2
tiles = [ch[x:x + H, y:y + W] for x in range(0, ch.shape[0], H) for y in range(0, ch.shape[1], W)]
x = np.array(tiles[1])
plt.imshow(x)

pix_2 = x * 65535.000
im_pil = Image.fromarray(np.uint16(pix_2))
fig_ch2 = px.imshow(im_pil, binary_string=True, binary_backend="jpg",width=500,height=500,title='Target:',binary_compression_level=9).update_xaxes(showticklabels = False).update_yaxes(showticklabels = False)
fig_ch3 = px.imshow(im_pil, binary_string=True, binary_backend="jpg",width=500,height=500,title='Target:',binary_compression_level=9).update_xaxes(showticklabels = False).update_yaxes(showticklabels = False)
ll = [fig_ch2]
ll.append(fig_ch3)
ll[0].show()


nuc_s = AIP
S_object.Nucleus_segmentation(img['1'], inv=False, for_dash=False,rescale_image=True )
seg = AIPS_object.Cytosol_segmentation(ch, ch2, nuc_s['sort_mask'], nuc_s['sort_mask_bin'], rescale_image=False)
# dict_ = {'img':img,'nuc':nuc_s,'seg':seg}
# try to work on img
nmask2 = nuc_s['nmask2']
nmask4 = nuc_s['nmask4']
sort_mask_bin = nuc_s['sort_mask_bin']
sort_mask_bin = skimage.transform.rescale(sort_mask_bin, 0.25, anti_aliasing=False)
sort_mask_bin = np.where(sort_mask_bin > 0, 1, 0)
sort_mask_bin = binary_erosion(sort_mask_bin, structure=np.ones((3, 3))).astype(np.float64)
print(np.unique(sort_mask_bin))
plt.imshow(sort_mask_bin)
sort_mask = nuc_s['sort_mask']
sort_mask = skimage.transform.rescale(sort_mask, 0.25, anti_aliasing=False)
x = np.where(sort_mask_bin > 0,sort_mask,0)
sort_mask = np.where(np.mod(x,1)>0,0,x)
sort_mask = np.array(sort_mask,np.uint32)
np.unique(sort_mask)
plt.imshow(sort_mask)

sort_mask = nuc_s['sort_mask']
sort_mask = np.where(np.mod(sort_mask,1)>0,0,sort_mask)
plt.imshow(sort_mask)

sort_mask = np.array(sort_mask,np.int)
np.unique(sort_mask)
plt.imshow(sort_mask)


x = np.array([1,1.2,3])
y = np.where(np.mod(x,1)>0,0,x)

sort_mask_bin_8 = np.array(sort_mask_bin, dtype=np.ubyte)
np.unique(sort_mask_bin_8)
plt.imshow(sort_mask_bin)
plt.imshow(sort_mask_bin_8)
sort_mask_bin_small = skimage.transform.rescale(sort_mask_bin, 0.25, anti_aliasing=False)
sort_mask_bin_small = np.where(sort_mask_bin_small > 0 ,1,0)
plt.imshow(sort_mask_bin_small)


plt.imshow(sort_mask)
np.unique(sort_mask)
sort_mask_small = skimage.transform.rescale(sort_mask, 0.25, anti_aliasing=False)
sort_mask_small = np.array(sort_mask_small, dtype=np.uint32)
plt.imshow(sort_mask_small)


#plt.imshow(sort_mask)
np.unique(sort_mask)


np.unique(sort_mask_bin)

np.ones_like(sort_mask)
sort_mask = np.array(sort_mask, dtype=np.uint8)
sort_mask_bin = np.array(sort_mask_bin, dtype=np.uint8)
np.unique(sort_mask)
np.unique(sort_mask_bin)

seg = AIPS_object.Cytosol_segmentation(ch, ch2, sort_mask, sort_mask_bin, rescale_image=True)

table = nuc_s['table']
cell_mask_1 = seg['cell_mask_1']
combine = seg['combine']
cseg_mask = seg['cseg_mask']
np.unique(cseg_mask)
plt.imshow(cseg_mask)
np.unique(cell_mask_1)
cseg_mask_bin = seg['cseg_mask_bin']
info_table = seg['info_table']
mask_unfiltered = seg['mask_unfiltered']
plt.imshow(mask_unfiltered)


input_gs_image = (ch / ch.max()) * 255
ch2_u8 = np.uint8(input_gs_image)
rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
rgb_input_img[:, :, 0] = ch2_u8
rgb_input_img[:, :, 1] = ch2_u8
rgb_input_img[:, :, 2] = ch2_u8


plt.imshow(mask_unfiltered)
a = plot_composite_image(rgb_input_img,mask_unfiltered,'test',alpha=0.2)
a.show()
im_pil_nmask2 = af.px_pil_figure(mask_unfiltered, bit=1, mask_name='nmask2', fig_title='Local threshold map - seed',wh=500)
im_pil_nmask2.show()

from PIL import Image
im1 = Image.open('/Users/kanferg/Desktop/Temp/1/c28.png').convert('L')
im2 = Image.open('/Users/kanferg/Desktop/Temp/1/i24.png').convert('L')



input_gs_image = (ch / ch.max()) * 255
ch2_u8 = np.uint8(input_gs_image)
rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
rgb_input_img[:, :, 0] = ch2_u8
rgb_input_img[:, :, 1] = ch2_u8
rgb_input_img[:, :, 2] = ch2_u8
img_gs = img_as_ubyte(rgb_input_img)
im_pil = Image.fromarray(img_gs)

img_nmask = img_as_ubyte(nmask2)
im_nmask = Image.fromarray(img_nmask).convert('RGB')
plt.imshow(im_nmask)

im3 = Image.blend(im_pil, im_nmask, 0.2)
plt.imshow(im3)

plt.imshow(sort_mask)
sort_mask = np.array(sort_mask,np.int32)
im_nmask = Image.fromarray(sort_mask).convert('RGB')
plt.imshow(im_nmask)
im3 = Image.blend(im_pil, im_nmask, 0.2)


im3 = Image.blend(im1, im2, 0.8)
# to show specified image
im3.show()

import skimage.segmentation as seg
import skimage.filters as filters


#################RGBA
input_gs_image = (ch / ch.max()) * 255
ch2_u8 = np.uint8(input_gs_image)
rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
rgb_input_img[:, :, 0] = ch2_u8
rgb_input_img[:, :, 1] = ch2_u8
rgb_input_img[:, :, 2] = ch2_u8
img_gs = img_as_ubyte(rgb_input_img)
im_pil = Image.fromarray(img_gs).convert('RGBA')
plt.imshow(im_pil)

sort_mask = nuc_s['sort_mask']
np.unique(sort_mask)
sort_mask = np.array(sort_mask,dtype=np.int32)
norm = (sort_mask - np.min(sort_mask))/(np.max(sort_mask) - np.min(sort_mask))
norm = norm*256
sort_mask = np.array(norm,dtype=np.uint8)
plt.imshow(sort_mask)

cm = plt.get_cmap('gist_rainbow')
colored_image = cm(sort_mask)
sort_mask = np.array(sort_mask,dtype=np.int32)
norm = (sort_mask - np.min(sort_mask))/(np.max(sort_mask) - np.min(sort_mask))
colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
plt.imshow(colored_image)


img_nmask = img_as_ubyte(colored_image)
im_nmask = Image.fromarray(img_nmask).convert('RGB')
plt.imshow(im_nmask)

img_gs = img_as_ubyte(rgb_input_img)
im_pil = Image.fromarray(img_gs).convert('RGB')

im3 = Image.blend(im_pil, im_nmask, 0.2)
plt.imshow(im3)


sort_mask = np.array(sort_mask,dtype=np.int32)
plt.imshow(sort_mask)
ch1_sort_mask = af.rgb_file_gray_scale(ch, mask=sort_mask, channel=0)

nmask2 = np.array(nmask2,dtype=np.int32)
plt.imshow(nmask2)
im_nmask = Image.fromarray(sort_mask).convert('RGBA')
plt.imshow(im_nmask)
im3 = Image.blend(im_pil, im_nmask, 0.5)
plt.imshow(im3)

info_table = pd.DataFrame(
            measure.regionprops_table(
                nmask2,
                intensity_image=ch,
                properties=['area', 'label', 'centroid'],
            )).set_index('label')
info_table.index.values
np.unique(sort_mask)



fig_im_pil_sort_mask = af.px_pil_figure(ch1_sort_mask, bit=3, mask_name='sort_mask', fig_title='RGB map - seed',wh=500)






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
