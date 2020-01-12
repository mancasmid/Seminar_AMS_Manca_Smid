import os
import numpy as np
import SimpleITK as itk
from tqdm import tqdm
from keras import backend as K

from os.path import join

def resample_image(input_image, spacing_mm=(1, 1, 1), spacing_image=None, inter_type=itk.sitkLinear):
    """
    Resample image to desired pixel spacing.

    Should specify destination spacing immediate value in parameter spacing_mm or as SimpleITK.Image in spacing_image.
    You must specify either spacing_mm or spacing_image, not both at the same time.

    :param input_image: Image to resample.
    :param spacing_mm: Spacing for resampling in mm given as tuple or list of two/three (2D/3D) float values.
    :param spacing_image: Spacing for resampling taken from the given SimpleITK.Image.
    :param inter_type: Interpolation type using one of the following options:
                            SimpleITK.sitkNearestNeighbor,
                            SimpleITK.sitkLinear,
                            SimpleITK.sitkBSpline,
                            SimpleITK.sitkGaussian,
                            SimpleITK.sitkLabelGaussian,
                            SimpleITK.sitkHammingWindowedSinc,
                            SimpleITK.sitkBlackmanWindowedSinc,
                            SimpleITK.sitkCosineWindowedSinc,
                            SimpleITK.sitkWelchWindowedSinc,
                            SimpleITK.sitkLanczosWindowedSinc
    :type input_image: SimpleITK.Image
    :type spacing_mm: Tuple[float]
    :type spacing_image: SimpleITK.Image
    :type inter_type: int
    :rtype: SimpleITK.Image
    :return: Resampled image as SimpleITK.Image.
    """
    resampler = itk.ResampleImageFilter()
    resampler.SetInterpolator(inter_type)

    if (spacing_mm is None and spacing_image is None) or \
       (spacing_mm is not None and spacing_image is not None):
        raise ValueError('You must specify either spacing_mm or spacing_image, not both at the same time.')

    if spacing_image is not None:
        spacing_mm = spacing_image.GetSpacing()

    input_spacing = input_image.GetSpacing()
    # set desired spacing
    resampler.SetOutputSpacing(spacing_mm)
    # compute and set output size
    output_size = np.array(input_image.GetSize()) * np.array(input_spacing) \
                  / np.array(spacing_mm)
    output_size = list((output_size + 0.5).astype('uint32'))
    output_size = [int(size) for size in output_size]
    resampler.SetSize(output_size)

    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())

    resampled_image = resampler.Execute(input_image)

    return resampled_image


def load_mri_brain_data(output_size=(64, 64), modalities=('t1', 'flair')):
    """
    Load data from '../mri-brain-slices' folder in the format suitable for training neural networks.

    This functions will load all images, perform cropping to fixed size and resample the obtained image such
    that the output size will match the one specified by parameter 'output_size'. The data output will contain the
    modalities as specified by parameter 'modalities'.

    :param output_size: Define output image size.
    :param modalities: List of modalities specified by strings 't1' and/or 'flair'.
    :type output_size: tuple[int]
    :type modalities: tuple[str]
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :return: Image data, brainmask and brain segmentation in a tuple.
    """
    # hidden function parameters
    DATA_PATH = './data'

    # define image extraction function based on cropping and resampling
    def extract_image(image, output_size=(128, 128), interpolation_type=itk.sitkLinear):
        new_spacing_mm = (192 / output_size[0], 192 / output_size[1], 1)
        return resample_image(
            itk.RegionOfInterest(image, (192, 192, 1), (0, 18, 0)), 
            spacing_mm = new_spacing_mm, 
            inter_type=interpolation_type)
    
    # load and extract all images and masks into a list of dicts
    mri_data = []
    patient_paths = os.listdir(DATA_PATH)
    for pacient_no in tqdm(range(len(patient_paths))):
        patient_path = join(DATA_PATH, patient_paths[pacient_no])

        # read all images
        t1 = itk.ReadImage(join(patient_path, 't1w.nii.gz'))
        flair = itk.ReadImage(join(patient_path, 'flair.nii.gz'))
        bmsk = itk.ReadImage(join(patient_path, 'brainmask.nii.gz'))
        seg = itk.ReadImage(join(patient_path, 'seg.nii.gz'))

        # crop and resample the images
        t1 = extract_image(t1, output_size, itk.sitkLinear) 
        flair = extract_image(flair, output_size, itk.sitkLinear)
        bmsk = extract_image(bmsk, output_size, itk.sitkNearestNeighbor)
        seg = extract_image(seg, output_size, itk.sitkNearestNeighbor)

        # add to dict
        mri_data.append({'t1':t1, 'flair':flair, 'bmsk':bmsk, 'seg':seg})
        
    # reshape all modalities and masks into 3d arrays
    t1_array = np.dstack([np.squeeze(itk.GetArrayFromImage(data['t1'])) for data in mri_data])
    flair_array = np.dstack([np.squeeze(itk.GetArrayFromImage(data['flair'])) for data in mri_data])
    bmsk_array = np.dstack([np.squeeze(itk.GetArrayFromImage(data['bmsk'])) for data in mri_data])
    seg_array = np.dstack([np.squeeze(itk.GetArrayFromImage(data['seg'])) for data in mri_data])
    
    # reshape the 3d arrays such that the number of cases is in the first column
    t1_array = np.transpose(t1_array, (2, 0, 1))
    flair_array = np.transpose(flair_array, (2, 0, 1))
    bmsk_array = np.transpose(bmsk_array, (2, 0, 1))
    seg_array = np.transpose(seg_array, (2, 0, 1))
     
    # reshape the 3d arrays according to the Keras backend
    if K.image_data_format() == 'channels_first': 
        # this format is (n_cases, n_channels, image_height, image_width)
        t1_karray = t1_array[:, np.newaxis, :, :]
        flair_karray = flair_array[:, np.newaxis, :, :]
        bmsk_karray = bmsk_array[:, np.newaxis, :, :]
        seg_karray = seg_array[:, np.newaxis, :, :]
        channel_axis = 1
    else:
        # this format is (n_cases, image_height, image_width, n_channels)
        t1_karray = t1_array[:, :, :, np.newaxis]
        flair_karray = flair_array[:, :, :, np.newaxis]
        bmsk_karray = bmsk_array[:, :, :, np.newaxis]
        seg_karray = seg_array[:, :, :, np.newaxis]
        channel_axis = -1

    if 'flair' in modalities and 't1' in modalities: 
        data = np.concatenate((t1_karray,flair_karray),axis=channel_axis)
    elif 't1' in modalities:
        data = t1_karray
    elif 'flair' in modalities:
        data = flair_karray
    else:
        raise ValueError('The input modalities "{}" are not recognized!'.format(MODALITIES))        
                
    # read image sizes and channel number
    _, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = data.shape

    # compute min and max values per each channel
    def stat_per_channel(values, stat_fcn):
        return stat_fcn(
            np.reshape(
                values, 
                (values.shape[0]*IMG_HEIGHT*IMG_WIDTH, IMG_CHANNELS)), 
            axis=0)[:, np.newaxis]

    min_data, max_data = stat_per_channel(data, np.min), stat_per_channel(data, np.max)
    min_data = np.reshape(min_data, (1, 1, 1, IMG_CHANNELS))
    max_data = np.reshape(max_data, (1, 1, 1, IMG_CHANNELS))

    # normalize image intensities to interval [0, 1]
    X = (data - min_data) / (max_data - min_data)

    # return the image modalities, brainmasks and brain segmentations
    return X, bmsk_karray, seg_karray