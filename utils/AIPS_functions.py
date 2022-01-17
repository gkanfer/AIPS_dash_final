'''
Function for AIPS DASH
'''
import xml.etree.ElementTree as xml
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


def segmentation_2ch(ch,ch2, rmv_object_nuc, block_size, offset,int_nuc, cyto_seg,
                     block_size_cyto,int_cyto ,offset_cyto, global_ther, rmv_object_cyto, rmv_object_cyto_small):
    '''
       Function for exploring the parameters for simple threshold based segmentation
       Prefer Nucleus segmentation
       Args:
           ch: Input image (tifffile image object)
           block_size: Detect local edges 1-99 odd
           offset: Detect local edges 0.001-0.9 odd
           rmv_object_nuc: percentile of cells to remove, 0.01-0.99
           cyto_seg: 1 or 0
           block_size_cyto: Detect local edges 1-99 odd
           offset_cyto: Detect local edges 0.001-0.9 odd
           global_ther: Percentile
           rmv_object_cyto:  percentile of cells to remove, 0.01-0.99
           rmv_object_cyto_small:  percentile of cells to remove, 0.01-0.99
       Returns:
            nmask2: local threshold binary map (eg nucleus)
            nmask4: local threshold binary map post opening (eg nucleus)
            sort_mask: RGB segmented image output first channel for mask (eg nucleus)
            cell_mask_2: local threshold binary map (eg cytoplasm)
            combine: global threshold binary map (eg cytoplasm)
            cseg_mask: RGB segmented image output first channel for mask (eg nucleus)
            test: Area table seed
            test2: Area table cytosol
    '''
    if int_nuc[0]==1:
        nmask = threshold_local(ch, block_size, "mean", np.median(np.ravel(ch))/10)
    else:
        nmask = threshold_local(ch, block_size, "mean", offset)
    nmask2 = ch > nmask
    nmask3 = binary_opening(nmask2, structure=np.ones((3, 3))).astype(np.float64)
    nmask4 = binary_fill_holes(nmask3)
    label_objects = sm.label(nmask4, background=0)
    info_table = pd.DataFrame(
        measure.regionprops_table(
            label_objects,
            intensity_image=ch,
            properties=['area', 'label','coords'],
        )).set_index('label')
    #info_table.hist(column='area', bins=100)
    test = info_table[info_table['area'] < info_table['area'].quantile(q=rmv_object_nuc)]
    sort_mask = label_objects
    if len(test) > 0:
        x = np.concatenate(np.array(test['coords']))
        sort_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
    else:
        test = info_table[info_table['area'] < info_table['area'].quantile(q=0.5)]
        x = np.concatenate(np.array(test['coords']))
        sort_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
    sort_mask_bin = np.where(sort_mask > 0, 1, 0)
    if cyto_seg[0]==1:
        if int_cyto[0]==1:
            ther_cell = threshold_local(ch2, block_size_cyto, "gaussian", np.median(np.ravel(ch2))/10)
        else:
            ther_cell = threshold_local(ch2, block_size_cyto, "gaussian", offset_cyto)
        cell_mask_1 = ch2 > ther_cell
        cell_mask_2 = binary_opening(cell_mask_1, structure=np.ones((3, 3))).astype(np.float64)
        quntile_num = np.quantile(ch2, global_ther)
        cell_mask_3 = np.where(ch2 > quntile_num, 1, 0)
        combine = cell_mask_2
        combine[cell_mask_3 > combine] = cell_mask_3[cell_mask_3 > combine]
        combine[sort_mask_bin > combine] = sort_mask_bin[sort_mask_bin > combine]
        cseg = watershed(np.ones_like(sort_mask_bin), sort_mask, mask=cell_mask_2)
        csegg = watershed(np.ones_like(sort_mask), cseg, mask=combine)
        info_table = pd.DataFrame(
            measure.regionprops_table(
                csegg,
                intensity_image=ch2,
                properties=['area', 'label','coords'],
            )).set_index('label')
       #info_table.hist(column='area', bins=100)
        ############# remove large object ################
        cseg_mask = csegg
        test1 = info_table[info_table['area'] > info_table['area'].quantile(q=rmv_object_cyto)]
        if len(test1) > 0:
            x = np.concatenate(np.array(test1['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        else:
            test1 = info_table[info_table['area'] > info_table['area'].quantile(q=0.5)]
            x = np.concatenate(np.array(test1['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        ############# remove small object ################
        test2 = info_table[info_table['area'] < info_table['area'].quantile(q=rmv_object_cyto_small)]
        if len(test2) > 0:
            x = np.concatenate(np.array(test2['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        else:
            test2 = info_table[info_table['area'] > info_table['area'].quantile(q=0.5)]
            x = np.concatenate(np.array(test2['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        dict = {'nmask2':nmask2, 'nmask4':nmask4, 'sort_mask':sort_mask,'cell_mask_1':cell_mask_1,
                'combine':cell_mask_3, 'cseg_mask':cseg_mask,'test':test, 'test2':test2}
        return dict
    else:
        dict = {'nmask2': nmask2, 'nmask4': nmask4,'test':test ,'sort_mask': sort_mask}
        return dict



def show_image_adjust(image, low_prec, up_prec):
    """
    image= np array 2d
    low/up precentile border of the image
    """
    percentiles = np.percentile(image, (low_prec, up_prec))
    scaled_ch1 = rescale_intensity(image, in_range=tuple(percentiles))
    return scaled_ch1
    # PIL_scaled_ch1 = Image.fromarray(np.uint16(scaled_ch1))
    # PIL_scaled_ch1.show()
    # return PIL_scaled_ch1

def save_pil_to_directory(img,bit,mask_name):
    '''
    :param img: image input
             bit:1 np.unit16 or 2 np.unit8
    :return: encoded_image (e_img)
    '''
    if bit == 1:
        # binary
        im_pil = Image.fromarray(img)
    elif bit == 2:
        # 16 image (normal image)
        im_pil = Image.fromarray(np.uint16(img))
    else:
        # ROI mask
        im_pil = Image.fromarray(np.uint8(img))
    filename1 = datetime.now().strftime("%Y%m%d_%H%M%S" + mask_name)
    im_pil.save(os.path.join('temp', filename1 + ".png"), format='png')  # this is for image processing
    e_img = base64.b64encode(open(os.path.join('temp', filename1 + ".png"), 'rb').read())
    return e_img


def XML_creat(filename,block_size,offset,rmv_object_nuc,block_size_cyto,offset_cyto,global_ther,rmv_object_cyto,rmv_object_cyto_small):
    root = xml.Element("Segment")
    cl = xml.Element("segment") #chiled
    root.append(cl)
    block_size_ = xml.SubElement(cl,"block_size")
    block_size_.text = block_size
    offset_ = xml.SubElement(cl,"offset")
    offset_.text = "13"
    rmv_object_nuc_ = xml.SubElement(cl, "rmv_object_nuc")
    rmv_object_nuc_.text = "rmv_object_nuc"
    block_size_cyto_ = xml.SubElement(cl, "block_size_cyto")
    block_size_cyto_.text = "block_size_cyto"
    offset_cyto_ = xml.SubElement(cl, "offset_cyto")
    offset_cyto_.text = "offset_cyto"
    global_ther_ = xml.SubElement(cl, "global_ther")
    global_ther_.text = "global_ther"
    rmv_object_cyto_ = xml.SubElement(cl, "rmv_object_cyto")
    rmv_object_cyto_.text = "rmv_object_cyto"
    rmv_object_cyto_small_ = xml.SubElement(cl, "rmv_object_cyto_small")
    rmv_object_cyto_small_.text = "rmv_object_cyto_small"
    tree = xml.ElementTree(root)
    with open(filename,'wb') as f:
        tree.write(f)

def seq(start, end, by=None, length_out=None):
    len_provided = True if (length_out is not None) else False
    by_provided = True if (by is not None) else False
    if (not by_provided) & (not len_provided):
        raise ValueError('At least by or n_points must be provided')
    width = end - start
    eps = pow(10.0, -14)
    if by_provided:
        if (abs(by) < eps):
            raise ValueError('by must be non-zero.')
        # Switch direction in case in start and end seems to have been switched (use sign of by to decide this behaviour)
        if start > end and by > 0:
            e = start
            start = end
            end = e
        elif start < end and by < 0:
            e = end
            end = start
            start = e
        absby = abs(by)
        if absby - width < eps:
            length_out = int(width / absby)
        else:
            # by is too great, we assume by is actually length_out
            length_out = int(by)
            by = width / (by - 1)
    else:
        length_out = int(length_out)
        by = width / (length_out - 1)
    out = [float(start)] * length_out
    for i in range(1, length_out):
        out[i] += by * i
    if abs(start + by * length_out - end) < eps:
        out.append(end)
    return out