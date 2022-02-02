
"""
Spyder Editor

This is a temporary script file.
"""

import inspect
import xml.etree.ElementTree as xml

#import skimage.color
import tifffile as tfi
import skimage.measure as sme
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
# from PIL import fromarray
from numpy import asarray
from skimage import data, io
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_opening
from skimage.morphology import disk, remove_small_objects
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank, gaussian, sobel
from skimage.util import img_as_ubyte
from skimage import data, util
from skimage.measure import regionprops_table
from skimage.measure import perimeter
from skimage import measure
from skimage.exposure import rescale_intensity, histogram
from skimage.feature import peak_local_max
import os
import glob
import pandas as pd
from pandas import DataFrame
from scipy.ndimage.morphology import binary_fill_holes
from skimage.viewer import ImageViewer
from skimage import img_as_float
import time
import base64
from datetime import datetime

from utils.controls import controls, controls_nuc, controls_cyto
from utils import AIPS_functions as af
from utils import AIPS_module as ai
from utils import display_and_xml as dx


path = '/Users/kanferg/Desktop/NIH_Youle/Python_projacts_general/dash/AIPS_Dash_Final/app_uploaded_files/'
#Composite.tif10.tif
AIPS_object = ai.Segment_over_seed(Image_name='dmsot0273_0003-512.tif', path=path, rmv_object_nuc=0.97, block_size=67,
                                           offset=0.000004,block_size_cyto=11, offset_cyto=-0.004, global_ther=0.3, rmv_object_cyto=0.99,
                                           rmv_object_cyto_small=0.43, remove_border=False)

# AIPS_object = ai.Segment_over_seed(Image_name='Composite.tif10.tif', path=path, rmv_object_nuc=0.9, block_size=59,
#                                            offset=0.0003,block_size_cyto=9, offset_cyto=0.0004, global_ther=0.4, rmv_object_cyto=0.99,
#                                            rmv_object_cyto_small=0.9, remove_border=False)

img = AIPS_object.load_image()
nuc_s = AIPS_object.Nucleus_segmentation(img['1'], inv=False)
seg = AIPS_object.Cytosol_segmentation(img['1'],img['0'],nuc_s['sort_mask'],nuc_s['sort_mask_bin'])


fig, ax = plt.subplots(1, 2, figsize=(12, 12))
ax[0].imshow(seg['sort_mask_sync'])
ax[1].imshow(seg['cseg_mask'])

np.unique(seg['cseg_mask'])
np.unique(seg['sort_mask_sync'])







ch2 = img['0']
ch = img['1']
bf_mask = dx.binary_frame_mask(ch,nuc_s['sort_mask'])
bf_mask = np.uint8(bf_mask)
plt.imshow(bf_mask)
roi_index_uni = np.unique(bf_mask)
roi_index_uni

ch2 = img['0']
ch2 = ch2*2**16
ch2 = np.uint8(ch2)
roi_index_uni = np.unique(ch2)
roi_index_uni
zero = np.zeros(np.shape(ch2),dtype=np.uint8)
#image_composite = np.stack((zero,ch2,bf_mask),dtype=np.uint8)
image_composite = np.zeros((np.shape(ch2)[0], np.shape(ch2)[1], 3), dtype=np.uint8)
image_composite[:,:,1] = np.where(ch2 > 0, ch2, 0)
image_composite[:,:,2] = np.where(bf_mask > 0, bf_mask, 0)
#image_composite = image_composite.reshape(np.shape(ch2)[0],np.shape(ch2)[1],3)
im_pil = Image.fromarray(image_composite, mode='RGB')
plt.imshow(im_pil)




ch2 = img['0']
ch2 = ch2*2**16
ch2 = np.uint8(ch2)
im_pil = Image.fromarray(ch2, mode='RGB')
plt.imshow(ch2)


ch2 = img['0']
ch2 = np.float16(ch2)


bf_mask = dx.binary_frame_mask(ch,nuc_s['sort_mask'])
bf_mask = np.uint8(bf_mask)
bf_mask = bf_mask/256
bf_mask = np.float16(bf_mask)
roi_index_uni = np.unique(bf_mask)
roi_index_uni

plt.imshow(bf_mask)
plt.imshow(ch2)
zero = np.zeros(np.shape(ch2),dtype=np.float16)

im1 = Image.fromarray(ch2,mode='L')
im2 = Image.fromarray(bf_mask,mode='L')
im_pil = Image.merge("HSV",(im1,im1,im2))
plt.imshow(im_pil)

image_composite = np.zeros((np.shape(ch2)[0], np.shape(ch2)[1], 3), dtype=np.float16)
image_composite[:,:,1] = ch2
image_composite[:,:,2] = bf_mask
im_pil = Image.fromarray(image_composite,mode='RGB')
plt.imshow(im_pil)



roi_index_uni = np.unique(img)
roi_index_uni = roi_index_uni[roi_index_uni > 1]
sort_mask_buffer = np.zeros((np.shape(img)[0], np.shape(img)[1], 3), dtype=np.uint8)
for npun in roi_index_uni:
    for i in range(3):
        sort_mask_buffer[img == npun, i] = unique_rand(2, 255, 1)[0]
im_pil = Image.fromarray(sort_mask_buffer, mode='RGB')
filename1 = datetime.now().strftime("%Y%m%d_%H%M%S" + mask_name)
im_pil.save(os.path.join(output_dir, filename1 + ".png"), format='png')




ch2 = img['0']
ch = img['1']
bf_mask = dx.binary_frame_mask(ch,nuc_s['sort_mask'])
bf_mask = np.uint8(bf_mask)
roi_index_uni = np.unique(bf_mask)
roi_index_uni

ch2 = img['0']
ch2 = ch2*2**16
ch2 = np.uint8(ch2)

ch2 = ch2.reshape(np.shape(ch2)[0],np.shape(ch2)[1],1)
bf_mask = bf_mask.reshape(np.shape(ch2)[0],np.shape(ch2)[1],1)
zero = np.zeros(np.shape(ch2),dtype=np.uint8)
zero[bf_mask > 0]=255

plt.imshow(zero)

image_composite = np.concatenate((zero,ch2,ch2),2)
im_pil = Image.fromarray(image_composite,mode='RGB')
plt.imshow(im_pil)

from utils import display_and_xml as dx

ch2 = img['0']
ch2 = ch2*2**16
ch2 = np.uint8(ch2)
roi_index_uni = np.unique(ch2)
roi_index_uni = roi_index_uni[roi_index_uni > 1]
im1 = np.ones((np.shape(ch2)[0], np.shape(ch2)[1], 1), dtype=np.uint8)
for npun in roi_index_uni:
    im1[ch2 == npun] = dx.unique_rand(2, 255, 1)[0]

im2 = np.ones((np.shape(ch2)[0], np.shape(ch2)[1], 1), dtype=np.uint8)
for npun in roi_index_uni:
    im2[ch2 == npun] = dx.unique_rand(2, 255, 1)[0]

im3 = np.zeros(np.shape(im1),dtype=np.uint8)
im3[bf_mask > 0]=255

image_composite = np.concatenate((im1,im2,im3),2)
im_pil = Image.fromarray(image_composite, mode='RGB')
plt.imshow(im_pil)



byteArray = bytearray(ch2)
grayImage = np.array( byteArray).reshape(512, 512)
plt.imshow(grayImage)

bf_mask = dx.binary_frame_mask(ch,nuc_s['sort_mask'])
bf_mask = np.uint8(bf_mask)

grayImage[grayImage==bf_mask,]=255
plt.imshow(grayImage)


#image_composite = np.zeros((np.shape(ch2)[0], np.shape(ch2)[1], 3), dtype=np.uint8)
image_composite = np.concatenate((ch2,ch2,ch2),2)
image_composite[bf_mask > 0 ,1] = 255
image_composite[:,:,2] = ch2.reshape(np.shape(ch2)[0],np.shape(ch2)[1],1)
image_composite[:,:,3] = ch2
plt.imshow(image_composite)



im_pil = Image.fromarray(image_composite,mode='RGB')
plt.imshow(im_pil)




ch2 = img['0']
ch2 = np.float16(ch2)


bf_mask = dx.binary_frame_mask(ch,nuc_s['sort_mask'])
bf_mask = np.uint8(bf_mask)
#bf_mask = bf_mask/256
# bf_mask = np.float16(bf_mask)
# roi_index_uni = np.unique(bf_mask)
# roi_index_uni


ch2 = img['0']
ch2 = ch2*2**16
ch2 = np.uint8(ch2)
#ch2 = ch2/256

im1 = Image.fromarray(ch2)
im1.gray()
im2 = Image.fromarray(plt.gray(bf_mask))
im_pil = Image.merge("HSV",(im1,im1,im2))
plt.imshow(im_pil)




ch2 = img['0']
ch2 = ch2*2**16
ch2 = np.uint8(ch2)
ch2_gray = np.uint8(np.where(ch2 > 230,255,ch2))
#ch2_gray = np.uint8(np.where(ch2 < 100,255,ch2))
plt.imshow(ch2)

from skimage import color
ch2 = img['0']
ch2_gray = color.gray2rgb(ch2)
plt.imshow(ch2_gray,cmap='grey')
# basic conversion from gray to RGB encoding
test_image = np.array([[[s,s,s] for s in r] for r in ch2_gray],dtype="u1")
plt.imshow(test_image,cmap='gray')
test_image = np.array([[[0,s,0] for s in r] for r in ch2_gray],dtype="u1")

ther_cell = threshold_local(img['0'], -0.0004, "gaussian", -0.0004)
blank = np.zeros(np.shape(img['0']))
cell_mask_1 = ch2 > ther_cell
cell_mask_2 = binary_opening(cell_mask_1, structure=np.ones((3, 3))).astype(np.float64)
quntile_num = np.quantile(ch2,0.2)
cell_mask_3 = np.where(ch2 > quntile_num, 1, 0)
combine = cell_mask_2
combine[cell_mask_3 > combine] = cell_mask_3[cell_mask_3 > combine]
combine[nuc_s['sort_mask_bin'] > combine] = nuc_s['sort_mask_bin'][nuc_s['sort_mask_bin'] > combine]
cseg = watershed(np.ones_like(nuc_s['sort_mask_bin']), nuc_s['sort_mask'], mask=cell_mask_2)
csegg = watershed(np.ones_like(nuc_s['sort_mask']), cseg, mask=combine)
plt.imshow(cseg)

info_table = pd.DataFrame(
    measure.regionprops_table(
        csegg,
        intensity_image=ch2,
        properties=['area', 'label', 'centroid', 'coords'],
    )).set_index('label')
# info_table.hist(column='area', bins=100)
############# remove large object ################
cseg_mask = csegg
plt.imshow(csegg)
table_unfiltered = info_table
test1 = info_table[info_table['area'] > info_table['area'].quantile(q=0.9)]
if len(test1) > 0:
    x = np.concatenate(np.array(test1['coords']))
    cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
    #info_table = info_table.drop(test1.index)
else:
    cseg_mask = cseg_mask
############# remove small object ################
test2 = info_table[info_table['area'] < info_table['area'].quantile(q=0.9)]
len(test2)
if len(test2) > 0:
    x = np.concatenate(np.array(test2['coords']))
    cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
else:
    cseg_mask = cseg_mask
# sync seed mask with cytosol mask
if self.remove_border:
    y_axis = np.shape(ch2)[0]
    x_axis = np.shape(ch2)[1]
    empty_array = np.zeros(np.shape(ch2))
    empty_array[0:1, 0:y_axis] = cseg_mask[0:1, 0:y_axis]  # UP
    empty_array[y_axis - 1:y_axis, 0:y_axis] = cseg_mask[y_axis - 1:y_axis, 0:y_axis]  # down
    empty_array[0:x_axis, 0:1] = cseg_mask[0:x_axis, 0:1]
    empty_array[0:x_axis, y_axis - 1:y_axis] = cseg_mask[0:x_axis, y_axis - 1:y_axis]  # left
    u, indices = np.unique(empty_array[empty_array > 0], return_inverse=True)  # u is unique values greater then zero
    remove_border_ = list(np.int16(u))
    for i in list(remove_border_):
        cseg_mask = np.where(cseg_mask == i, 0, cseg_mask)
    info_table = pd.DataFrame(
        measure.regionprops_table(
            cseg_mask,
            intensity_image=ch2,
            properties=['area', 'label', 'centroid'],
        )).set_index('label')
else:
    if len(info_table) > 1:
        info_table = pd.DataFrame(
            measure.regionprops_table(
                cseg_mask,
                intensity_image=ch2,
                properties=['area', 'label', 'centroid'],
            )).set_index('label')
    else:
        dict_blank = {'area': [0, 0], 'label': [0, 0], 'centroid': [0, 0]}
        info_table = pd.DataFrame(dict_blank)
info_table['label'] = range(2, len(info_table) + 2)
# round
info_table = info_table.round({'centroid-0': 0, 'centroid-1': 0})
info_table = info_table.reset_index(drop=True)
sort_mask_bin = np.where(sort_mask > 0, 1, 0)
cseg_mask_bin = np.where(cseg_mask > 0, 1, 0)
combine_namsk = np.where(sort_mask_bin + cseg_mask_bin > 1, sort_mask, 0)
# test masks if blank then return blank
cell_mask_1 = evaluate_image_output(cell_mask_1)
# combine = evaluate_image_output(combine)
combine_namsk = evaluate_image_output(combine_namsk)
cseg_mask = evaluate_image_output(cseg_mask)
# check data frame
if len(info_table) == 0:
    d = {'area': [0], 'centroid-0': [0], 'centroid-1': [0], 'label': [0]}
    info_table = pd.DataFrame(d)
else:
    info_table = info_table







#
# fig, ax = plt.subplots(2, 3, figsize=(12, 12))
# ax[0][0].imshow(nuc_s['nmask2'], cmap=plt.cm.gray)
# ax[0][1].imshow(nuc_s['nmask4'], cmap=plt.cm.gray)
# ax[0][2].imshow(nuc_s['sort_mask'], cmap=plt.cm.gray)

#
# fig, ax = plt.subplots(2, 3, figsize=(12, 12))
# ax[0][0].imshow(seg['cell_mask_1'], cmap=plt.cm.gray)
# ax[0][1].imshow(seg['cseg_mask'], cmap=plt.cm.gray)
# ax[0][2].imshow(seg['mask_unfiltered'], cmap=plt.cm.gray)


