from amslib import resample_image
import numpy as np
import SimpleITK as itk
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


def img2array(img):
    return np.squeeze(itk.GetArrayFromImage(img))


def extract_image(image, output_size=(128, 128), interpolation_type=itk.sitkLinear, extraction_index=(0, 0, 0), input_size=(192, 192, 1)):
    new_spacing_mm = (input_size[0] / output_size[0], input_size[1] / output_size[1], 1)
    return resample_image(
        itk.RegionOfInterest(image, input_size, extraction_index),
        spacing_mm=new_spacing_mm,
        inter_type=interpolation_type)


def stat_per_channel(values, stat_fcn, img_height, img_width, img_channels):
    return stat_fcn(
        np.reshape(
            values,
            (values.shape[0]*img_height*img_width, img_channels)),
        axis=0)[:, np.newaxis]


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def iou_coef(y_true, y_pred, smooth=1):
    """
    IoU = (|X &amp; Y|)/ (|X or Y|)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    return (intersection + smooth) / (union + smooth)


def iou_coef_loss(y_true, y_pred):
    return -iou_coef(y_true, y_pred)


def dice_coef(y_true, y_pred):
    """
    DSC = (2*|X &amp; Y|)/ (|X| + |Y|)
    """
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def get_mri_data(sct_results_path, output_size, number_of_used_patients=28, plane='transversal', extraction_index=(0,0,0), img_input_size=(192, 192, 1)):
    """
    Vrne podatke MRI pacientov za izbrano ravnino.

    :param sct_results_path: Pot do mape 'sct_results'.
    :param output_size: Izhodna velikost dobljenih slik.
    :param number_of_used_patients: Število pacientov, ki bodo uporabljeni pri pridobivanju slik.
    :param plane: Želena ravnina, lahko je 'transversal', 'sagital' ali 'coronal'.
    :param extraction_index: Indeks na sliki za področje zanimanja.
    :param img_input_size: Vhodna velikost slik.
    :return: t1_array, seg_array, patien_no_array, kjer je t1_array matrika 2D slik, seg_array so slike razgrajene
             hrbtenjače, patient_no_array je vektor, ki predstavlja indekse pacientov, katerim pripadajo slike.
    """
    if plane == 'transversal':
        mri_axis = 0
    elif plane == 'coronal':
        mri_axis = 1
    elif plane == 'sagital':
        mri_axis = 2
    else:
        raise ValueError("Parameter 'plane' can either be 'transversal', 'sagital' or 'coronal'")

    # za vsakega pacienta dobi slike T1 in razgradnje za želeno ravnino
    mri_data = []
    for patient_no in tqdm(range(1, number_of_used_patients + 1)):
        patient_path = os.path.join(sct_results_path, "sct-MS{}-results".format(patient_no))

        # preberi slike
        t1 = itk.ReadImage(os.path.join(patient_path, 'MS{}.nii.gz'.format(patient_no)))
        seg = itk.ReadImage(os.path.join(patient_path, 'MS{}_seg_labeled.nii.gz'.format(patient_no)))

        t1_arr = np.squeeze(itk.GetArrayFromImage(t1))
        seg_arr = np.squeeze(itk.GetArrayFromImage(seg))

        # Zberi vse rezine v matriko
        for img_slice in range(t1_arr.shape[mri_axis]):
            # glede na podano ravnino določi indeks, po katerem se bodo zbirale rezine
            idx = [np.s_[:], np.s_[:], np.s_[:]]
            idx[mri_axis] = img_slice

            t1_slice = t1_arr[tuple(idx)]
            t1_slice = t1_slice[np.newaxis, :, :]
            seg_slice = seg_arr[tuple(idx)]
            seg_slice = seg_slice[np.newaxis, :, :]

            # obreži slike, da bodo ustrezne velikosti
            t1_slice_ext = extract_image(itk.GetImageFromArray(t1_slice), output_size, itk.sitkLinear, extraction_index, img_input_size)
            seg_slice_ext = extract_image(itk.GetImageFromArray(seg_slice), output_size, itk.sitkLinear, extraction_index, img_input_size)

            # ce hrbtenjace v tej rezini ni, nadaljuj z naslednjo
            t1_max = itk.GetArrayFromImage(t1_slice_ext).max()
            t1_min = itk.GetArrayFromImage(t1_slice_ext).min()
            if not np.any(itk.GetArrayFromImage(seg_slice_ext)) or ((t1_max - t1_min) == 0):
                continue

            mri_data.append({'t1': t1_slice_ext, 'seg': seg_slice_ext, 'patient_no': patient_no})

    # zloži slike in maske v 3d polje
    t1_array = np.dstack([np.squeeze(itk.GetArrayFromImage(data['t1'])) for data in mri_data])
    seg_array = np.dstack([np.squeeze(itk.GetArrayFromImage(data['seg'])) for data in mri_data])

    patien_no_array = np.dstack([data['patient_no'] for data in mri_data])

    # preoblikuj polje tako, da je število vzorcev v prvem stolpcu
    t1_array = np.transpose(t1_array, (2, 0, 1))
    seg_array = np.transpose(seg_array, (2, 0, 1))

    return t1_array, seg_array, patien_no_array


def normalize_images(image_array):
    """
    Normalizira slike, da so vse vrednosti pikslov med 0 in 1.

    :param image_array: Vhodna matrika slik.
    :return: Matrika normaliziranih slik.
    """
    val_min = image_array.min(axis=1).min(axis=1)
    val_max = image_array.max(axis=1).max(axis=1)

    num_images = image_array.shape[0]
    val_min = np.reshape(val_min, (num_images, 1, 1))
    val_max = np.reshape(val_max, (num_images, 1, 1))

    image_array = (image_array - val_min) / (val_max - val_min)

    return image_array


def analysisBlandAltman(iBiomarker, iData1, iData2, iAxes=None):
    if iAxes is None:
        plt.figure()
        iAxes = plt.gca()

    iData1    = np.asarray(iData1)
    iData2    = np.asarray(iData2)
    mean      = np.mean([iData1, iData2], axis=0)
    diff      = iData1 - iData2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    iAxes.scatter(mean, diff)
    iAxes.axhline(0,            color='gray', linestyle='-',  linewidth=1)
    iAxes.axhline(md,           color='red',  linestyle='--', linewidth=2)
    iAxes.axhline(md + 1.96*sd, color='gray', linestyle='--', linewidth=2)
    iAxes.axhline(md - 1.96*sd, color='gray', linestyle='--', linewidth=2)
    iAxes.set_xlabel('Mean value')
    iAxes.set_ylabel('Difference')
    iAxes.set_title('{bname}, Bland-Altman (md={md:.2f}, sd={sd:.2f})'.format(
        bname=iBiomarker.upper(), md=md, sd=sd))


def read_csa_data_from_file(file_path):
    """
    Izlušči indekse rezin in pripadajoče CSA iz tekstovne datoteke csa_per_slice.txt.

    :param file_path: Pot do datoteke csa_per_slice.txt
    :return: Matriko Nx2, kjer je N število rezin. Prvi stolpec so indeksi rezin v transverzalni ravnini,
             drugi stolpec pa so pripadajoče vrednosti CSA.
    """
    with open(file_path, 'r') as f:
        content = f.readlines()

    csa_data = []
    for line in content[1:]:
        csa_data.append([float(val) for val in line.split(',')[:2]])

    return np.array(csa_data)


def load_sct_csa_data(sct_results_path, number_of_used_patients=28):
    """
    Dobi referenčne CSA podatke za paciente.

    :param sct_results_path: Pot do 'sct_results' mape.
    :param number_of_used_patients: Število uporabljenih pacientov.
    :return: Referenčni CSA podatki v obliki liste.
    """
    csa_data_list = []
    for patient_ind in range(1, number_of_used_patients+1):
        csa_data = read_csa_data_from_file(os.path.join(os.path.join(sct_results_path, "sct-MS{}-results".format(patient_ind)), "csa_per_slice.txt"))
        csa_data_list.append(csa_data)

    return csa_data_list
