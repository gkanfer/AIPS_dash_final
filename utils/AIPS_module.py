import tifffile as tfi
import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from scipy.ndimage.morphology import binary_opening, binary_erosion
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import measure
import os
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
from utils.display_and_xml import evaluate_image_output

class AIPS:
    def __init__(self, Image_name, path, rmv_object_nuc, block_size, offset):
        self.Image_name = Image_name
        self.path = path
        self.rmv_object_nuc = rmv_object_nuc
        self.block_size = block_size
        self.offset = offset

    def load_image(self):
        ''':parameter
        Image: File name (tif format) - should be greyscale
        path: path to the file
        :return
        grayscale_image_container: dictionary of np array
        '''
        grayscale_image_container = {}
        pixels = tfi.imread(os.path.join(self.path,self.Image_name))
        pixels_float = pixels.astype('float64')
        pixels_float = pixels_float / 65535.000
        if np.shape(pixels_float)[2]==2:
            pixels_float = pixels_float.reshape(np.shape(pixels_float)[2],np.shape(pixels_float)[0],np.shape(pixels_float)[1])
        if np.shape(pixels_float)[0] > 4:
            # single chanlle image
            grayscale_image_container = {'ch':pixels_float}
        else:
            for i in range(np.shape(pixels_float)[0]):
                dict = {'{}'.format(i):pixels_float[i, :, :]}
                grayscale_image_container.update(dict)
        return grayscale_image_container

    def get_name_dict(dict):
        '''
        dict: dictionary of np array
        :return
        l: list of name of the dictionary from load_image function
        '''
        l = []
        for name,dict_ in dict.items():
            l.append(name)
        return l

    def Nucleus_segmentation(self,ch,inv=False):
        '''
        ch: Input image (tifffile image object)
        ivt: if invert than no need to fill hall and open
        block_size: Detect local edges 1-99 odd
        offset: Detect local edges 0.001-0.9 odd
        rmv_object_nuc: percentile of cells to remove, 0.01-0.99
        :return:
        nmask2: local threshold binary map (eg nucleus)
        nmask4: local threshold binary map post opening (eg nucleus)
        sort_mask: RGB segmented image output first channel for mask (eg nucleus)
        sort_mask_bin: Binary
        '''
        nmask = threshold_local(ch, self.block_size, "mean", self.offset)
        blank = np.zeros(np.shape(ch))
        nmask2 = ch > nmask
        if inv:
            nmask2 = np.invert(nmask2)
            nmask4 = nmask2
        else:
            nmask3 = binary_opening(nmask2, structure=np.ones((3, 3))).astype(np.float64)
            nmask4 = binary_fill_holes(nmask3)
        label_objects = sm.label(nmask4, background=0)
        info_table = pd.DataFrame(
            measure.regionprops_table(
                label_objects,
                intensity_image=ch,
                properties=['area', 'label','coords','centroid'],
            )).set_index('label')
        #info_table.hist(column='area', bins=100)
        # remove small objects - test data frame of small objects
        test = info_table[info_table['area'] < info_table['area'].quantile(q=self.rmv_object_nuc)]
        sort_mask = label_objects
        if len(test) > 0:
            x = np.concatenate(np.array(test['coords']))
            sort_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        else:
            test = info_table[info_table['area'] < info_table['area'].quantile(q=0.5)]
            x = np.concatenate(np.array(test['coords']))
            sort_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        sort_mask_bin = np.where(sort_mask > 0, 1, 0)
        dict = {'nmask2':nmask2,'nmask4':nmask4,'sort_mask':sort_mask,'sort_mask_bin':sort_mask_bin, 'table':test}
        return dict

class Segment_over_seed(AIPS):
    def __init__(self, Image_name, path, rmv_object_nuc, block_size, offset,
                block_size_cyto, offset_cyto ,global_ther,rmv_object_cyto,rmv_object_cyto_small
                ,remove_border):
        super().__init__(Image_name, path, rmv_object_nuc, block_size, offset)
        self.block_size_cyto = block_size_cyto
        self.offset_cyto = offset_cyto
        self.global_ther = global_ther
        self.rmv_object_cyto = rmv_object_cyto
        self.rmv_object_cyto_small = rmv_object_cyto_small
        self.remove_border = remove_border

    def test_subclass(self):
        print(self)

    def Cytosol_segmentation(self, ch, ch2,sort_mask,sort_mask_bin):
        '''
        ch: Input image (tifffile image object)
        ch2: Input image (tifffile image object)
        sort_mask: RGB segmented image output first channel for mask (eg nucleus)
        sort_mask_bin: Binary
        block_size_cyto: Detect local edges 1-99 odd
        offset_cyto: Detect local edges 0.001-0.9 odd
        global_ther: Percentile
        rmv_object_cyto:  percentile of cells to remove, 0.01-0.99
        rmv_object_cyto_small:  percentile of cells to remove, 0.01-0.99
        remove_border: boolean -  object on border of image to be removed
        :return:
        nmask2: local threshold binary map (eg nucleus)
        nmask4: local threshold binary map post opening (eg nucleus)
        sort_mask: RGB segmented image output first channel for mask (eg nucleus)
        cell_mask_2: local threshold binary map (eg cytoplasm)
        combine: global threshold binary map (eg cytoplasm)
        sort_mask_syn: RGB segmented image output first channel for mask (eg nucleus) sync
        cseg_mask: RGB segmented image output first channel for mask (eg nucleus)
        test: Area table seed
        info_table: Area table cytosol synchronize
        '''
        ther_cell = threshold_local(ch2, self.block_size_cyto, "gaussian", self.offset_cyto)
        cell_mask_1 = ch2 > ther_cell
        cell_mask_2 = binary_opening(cell_mask_1, structure=np.ones((3, 3))).astype(np.float64)
        quntile_num = np.quantile(ch2, self.global_ther)
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
                properties=['area', 'label', 'centroid','coords'],
            )).set_index('label')
        # info_table.hist(column='area', bins=100)
        ############# remove large object ################
        cseg_mask = csegg
        test1 = info_table[info_table['area'] > info_table['area'].quantile(q = self.rmv_object_cyto)]
        if len(test1) > 0:
            x = np.concatenate(np.array(test1['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0

        else:
            test1 = info_table[info_table['area'] > info_table['area'].quantile(q=0.5)]
            x = np.concatenate(np.array(test1['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        ############# remove small object ################
        test2 = info_table[info_table['area'] < info_table['area'].quantile(q = self.rmv_object_cyto_small)]
        if len(test2) > 0:
            x = np.concatenate(np.array(test2['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        else:
            test2 = info_table[info_table['area'] > info_table['area'].quantile(q=0.5)]
            x = np.concatenate(np.array(test2['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        # sync seed mask with cytosol mask
        if self.remove_border:
            y_axis = np.shape(ch2)[0]
            x_axis = np.shape(ch2)[1]
            empty_array = np.zeros(np.shape(ch2))
            empty_array[0:1, 0:y_axis] = cseg_mask[0:1, 0:y_axis]  # UP
            empty_array[y_axis - 1:y_axis, 0:y_axis] = cseg_mask[y_axis - 1:y_axis, 0:y_axis]  # down
            empty_array[0:x_axis, 0:1] = cseg_mask[0:x_axis, 0:1]
            empty_array[0:x_axis, y_axis - 1:y_axis] = cseg_mask[0:x_axis, y_axis - 1:y_axis]  # left
            u, indices = np.unique(empty_array[empty_array > 0], return_inverse=True) #u is unique values greater then zero
            remove_border_ = list(np.int16(u))
            for i in list(remove_border_):
                cseg_mask = np.where(cseg_mask == i, 0, cseg_mask)
            info_table = pd.DataFrame(
                measure.regionprops_table(
                    cseg_mask,
                    intensity_image=ch2,
                    properties=['area', 'label', 'centroid'],
                )).set_index('label')
        else :
            info_table = pd.DataFrame(
                measure.regionprops_table(
                    cseg_mask,
                    intensity_image=ch2,
                    properties=['area', 'label', 'centroid'],
                )).set_index('label')
        info_table['label'] = range(2, len(info_table) + 2)
        # round
        info_table = info_table.round({'centroid-0': 0, 'centroid-1': 0})
        info_table = info_table.reset_index(drop=True)
        sort_mask_bin = np.where(sort_mask > 0, 1, 0)
        cseg_mask_bin = np.where(cseg_mask > 0, 1, 0)
        combine_namsk = np.where(sort_mask_bin + cseg_mask_bin > 1, sort_mask, 0)
        # test masks if blank then return blank
        cell_mask_1 = evaluate_image_output(cell_mask_1)
        #combine = evaluate_image_output(combine)
        combine_namsk = evaluate_image_output(combine_namsk)
        cseg_mask = evaluate_image_output(cseg_mask)
        # check data frame
        if len(info_table)==0:
            d = {'area': [0], 'centroid-0': [0], 'centroid-1': [0], 'label': [0]}
            info_table = pd.DataFrame(d)
        else:
            info_table=info_table
        dict = {'cell_mask_1': cell_mask_2,'combine': cell_mask_3,'sort_mask_sync':combine_namsk, 'cseg_mask': cseg_mask,'info_table': info_table}
        return dict