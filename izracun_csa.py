from keras.models import load_model
from keras.utils import to_categorical
from funkcije import *
import os
import matplotlib.pyplot as plt
import numpy as np


SCT_RESULTS_PATH = r'.\Gradivo\sct_results'
OUTPUT_SIZE_TRANS = (128, 128)
OUTPUT_SIZE_SAG = (256, 64)
NUMBER_OF_USED_PATIENTS = 28

# Nalozi dobljena modela
model_trans = load_model(os.path.join('models', 'model-mri-unet-seg-transversal-best.h5'),
                         custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})
model_sag = load_model(os.path.join('models', 'model-mri-unet-seg-sagital-best.h5'),
                       custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})

t1_trans_array, seg_trans_array, patient_no_trans_array = get_mri_data(SCT_RESULTS_PATH, OUTPUT_SIZE_TRANS, NUMBER_OF_USED_PATIENTS, 'transversal', (30, 30, 0))
t1_sag_array, seg_sag_array, patient_no_sag_array = get_mri_data(SCT_RESULTS_PATH, OUTPUT_SIZE_SAG, NUMBER_OF_USED_PATIENTS, 'sagital', img_input_size=(256, 64, 1))

# normaliziraj t1 slike
t1_trans_array = normalize_images(t1_trans_array)
t1_sag_array = normalize_images(t1_sag_array)

# segmentirano hrbtenjaco v sagitalni ravnini pretvori v kategoricne podatke
NUM_CLASSES = 7
seg_sag_array = to_categorical(seg_sag_array, NUM_CLASSES)

# pri opazovanju transverzalne ravnine nas zanima samo maska hrbtenjace
SPINAL_CORD = 1
seg_trans_array[seg_trans_array > 0] = SPINAL_CORD

# dodaj dimenzijo modalitet
if K.image_data_format() == 'channels_first':
    t1_trans_karray = t1_trans_array[:, np.newaxis, :, :]
    seg_trans_karray = seg_trans_array[:, np.newaxis, :, :]
    t1_sag_karray = t1_sag_array[:, np.newaxis, :, :]
    seg_sag_karray = seg_sag_array[:, np.newaxis, :, :]
else:
    t1_trans_karray = t1_trans_array[:, :, :, np.newaxis]
    seg_trans_array = seg_trans_array[:, :, :, np.newaxis]
    t1_sag_karray = t1_sag_array[:, :, :, np.newaxis]
    seg_sag_karray = seg_sag_array[:, :, :, np.newaxis]


preds_trans_data = model_trans.predict(t1_trans_karray, verbose=1)
preds_sag_data = model_sag.predict(t1_sag_karray, verbose=1)

preds_trans_data = (preds_trans_data.round()).astype(np.uint8)
preds_sag_data = (preds_sag_data.round()).astype(np.uint8)

trans_image_ind_to_show = 20

f, ax = plt.subplots(3, 1, sharex=True, sharey=True)

ax[0].imshow(t1_trans_karray[trans_image_ind_to_show, :, :, 0], cmap='gray')
ax[0].set_title("Vhodna slika")
ax[0].axis('off')

ax[1].imshow(seg_trans_array[trans_image_ind_to_show, :, :, 0], cmap='gray')
ax[1].set_title("Maska hrbtenjace - referenčna razgradnja")
ax[1].axis('off')

ax[2].imshow(preds_trans_data[trans_image_ind_to_show, :, :, 0], cmap='gray')
ax[2].set_title("Maska hrbtenjace - izhod Unet")
ax[2].axis('off')

sag_image_ind_to_show = 10
C2_INDEX = 2
C3_INDEX = 3
SPINAL_CORD_INDEX = 0

f, ax = plt.subplots(3, 2, sharex=True, sharey=True)
ax[0][0].imshow(t1_sag_karray[sag_image_ind_to_show, :, :, 0], cmap='gray')
ax[0][0].set_title("Vhodna slika")
ax[0][0].axis('off')

ax[0][1].axis('off')

ax[1][0].imshow(seg_sag_array[sag_image_ind_to_show, :, :, C2_INDEX], cmap='gray')
ax[1][0].set_title("Maska C2 - Referenčna razgradnja")
ax[1][0].axis('off')

ax[1][1].imshow(seg_sag_array[sag_image_ind_to_show, :, :, C3_INDEX], cmap='gray')
ax[1][1].set_title("Maska C3 - Referenčna razgradnja")
ax[1][1].axis('off')

ax[2][0].imshow(preds_sag_data[sag_image_ind_to_show, :, :, C2_INDEX], cmap='gray')
ax[2][0].set_title("Maska C2 - izhod Unet")
ax[2][0].axis('off')

ax[2][1].imshow(preds_sag_data[sag_image_ind_to_show, :, :, C3_INDEX], cmap='gray')
ax[2][1].set_title("Maska C3 - izhod Unet")
ax[2][1].axis('off')

# pretvori slike sagitalne ravnine v listo za vsakega pacienta
sagital_images_patient_list = []
transverzal_images_patient_list = []
for patient_ind in range(1, NUMBER_OF_USED_PATIENTS+1):
    preds_sag_patient_image = preds_sag_data[np.squeeze(patient_no_sag_array == patient_ind), :, :, :]
    preds_trans_patient_image = preds_trans_data[np.squeeze(patient_no_trans_array == patient_ind), :, :, :]

    sagital_images_patient_list.append(preds_sag_patient_image)
    transverzal_images_patient_list.append(preds_trans_patient_image)

# kategoriziraj rezine v transverzalni ravnini, za vsakega pacienta
slice_category_patient_list = []
for sag_image_data in sagital_images_patient_list:

    # najdi rezino, kjer je povrsina hrbtenjace najvecja
    preds_spinal_cord = sag_image_data[:, :, :, SPINAL_CORD_INDEX] - 1  # invertiraj
    spine_img_ind = np.argmax(np.sum(np.sum(preds_spinal_cord, axis=1), axis=1))

    # dobi maski c2 in c3
    c2_img = sag_image_data[spine_img_ind, :, :, C2_INDEX]
    c3_img = sag_image_data[spine_img_ind, :, :, C3_INDEX]

    # kategoriziraj vsako rezino v skupine: 0 - ne pripada segmentu, 1 - pripada C2, 2 - pripada C3
    slice_category_arr = np.argmax(np.vstack((np.zeros_like(c2_img[:, 0]), c2_img.sum(axis=1), c3_img.sum(axis=1))), axis=0)

    slice_category_patient_list.append(slice_category_arr)

for patient_ind in range(NUMBER_OF_USED_PATIENTS):
    trans_image = transverzal_images_patient_list[patient_ind][:, :, :, 0]

    temp_slice_vec = np.zeros((trans_image.shape[0], 1, 1))
    category_slice = slice_category_patient_list[patient_ind]

    if temp_slice_vec.shape[0] >= category_slice.shape[0]:
        temp_slice_vec[:category_slice.shape[0], 0, 0] = category_slice
    else:
        temp_slice_vec[:, 0, 0] = category_slice[:temp_slice_vec.shape[0]]

    trans_image = trans_image * temp_slice_vec

    # zamenjaj rezino z kategorizirano
    transverzal_images_patient_list[patient_ind] = trans_image

c2_num_pix_patient_list = []
c3_num_pix_patient_list = []
for patient_ind in range(NUMBER_OF_USED_PATIENTS):
    # dobi vektor stevil pikslov C2 in C3, ki pripadajo posamezni rezini
    c2_num_pix_arr = np.sum(np.count_nonzero(transverzal_images_patient_list[patient_ind] == 1, axis=1), axis=1)
    c3_num_pix_arr = np.sum(np.count_nonzero(transverzal_images_patient_list[patient_ind] == 2, axis=1), axis=1)

    c2_num_pix_patient_list.append(c2_num_pix_arr)
    c3_num_pix_patient_list.append(c3_num_pix_arr)

np.savez('csa_data.npz', c2_num_pix_patient_list=c2_num_pix_patient_list, c3_num_pix_patient_list=c3_num_pix_patient_list, NUMBER_OF_USED_PATIENTS=NUMBER_OF_USED_PATIENTS)

plt.show()
