import openpyxl as xl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from funkcije import load_sct_csa_data, analysisBlandAltman

csa_data = np.load('csa_data.npz', allow_pickle=True)

c2_num_pix_patient_list = csa_data['c2_num_pix_patient_list']
c3_num_pix_patient_list = csa_data['c3_num_pix_patient_list']
NUMBER_OF_USED_PATIENTS = csa_data['NUMBER_OF_USED_PATIENTS']

sct_csa_data = load_sct_csa_data(r".\Gradivo\sct_results", NUMBER_OF_USED_PATIENTS)

# zberi podatke v listo, odstrani rezine, ki ne vsebujejo c2 ali c3
csa2_preds_lst, csa2_ref_lst, csa3_preds_lst, csa3_ref_lst = [], [], [], []
for patient_ind in range(NUMBER_OF_USED_PATIENTS):
    # v sct_csa_data uporabi samo toliko elementov, kolikor jih je tudi v c2 in c3
    sct_csa_data[patient_ind] = sct_csa_data[patient_ind][:c2_num_pix_patient_list[patient_ind].shape[0], 1]

    csa2_preds_lst.append(c2_num_pix_patient_list[patient_ind][c2_num_pix_patient_list[patient_ind] != 0])
    csa2_ref_lst.append(sct_csa_data[patient_ind][c2_num_pix_patient_list[patient_ind] != 0])
    csa3_preds_lst.append(c3_num_pix_patient_list[patient_ind][c3_num_pix_patient_list[patient_ind] != 0])
    csa3_ref_lst.append(sct_csa_data[patient_ind][c3_num_pix_patient_list[patient_ind] != 0])

# preberi podatke iz patient_data.xlsx, da dobiš EDSS za vsakega pacienta
wb = xl.load_workbook('Gradivo\\patient_data\\patient_data.xlsx')
ws = wb['Data']

patient_data = []
patient_keys = [('A', 'id'), ('B', 'age'), ('C', 'trajanje_bolezni'), ('D', 'spol'), ('E', 'edss')]
for i in range(1, NUMBER_OF_USED_PATIENTS + 1):
    patient_data.append({patient_keys[j][1]: ws[patient_keys[j][0]][i].value for j in range(len(patient_keys))})

edss_data = np.array([patient_data[i]['edss'] for i in range(NUMBER_OF_USED_PATIENTS)])

# dobi povprečje CSA C2/C3 za vsakega pacienta
CSA_C2_preds_mean_lst, CSA_C3_preds_mean_lst, CSA_C2_ref_mean_lst, CSA_C3_ref_mean_lst = [], [], [], []
for patient_ind in range(NUMBER_OF_USED_PATIENTS):
    CSA_C2_preds_mean_lst.append(np.mean(csa2_preds_lst[patient_ind]))
    CSA_C3_preds_mean_lst.append(np.mean(csa3_preds_lst[patient_ind]))

    CSA_C2_ref_mean_lst.append(np.mean(csa2_ref_lst[patient_ind]))
    CSA_C3_ref_mean_lst.append(np.mean(csa3_ref_lst[patient_ind]))

print("Korelacija med CSA2 in EDSS (izračunane vrednosti CSA): "
      "Spearmanov rank: {:.4f}, p-vrednost: {:.4f}".format(*list(stats.spearmanr(edss_data, CSA_C2_preds_mean_lst))))
print("Korelacija med CSA3 in EDSS (izračunane vrednosti CSA): "
      "Spearmanov rank: {:.4f}, p-vrednost: {:.4f}".format(*list(stats.spearmanr(edss_data, CSA_C3_preds_mean_lst))))

print("Korelacija med CSA2 in EDSS (referenčne vrednosti CSA): "
      "Spearmanov rank: {:.4f}, p-vrednost: {:.4f}".format(*list(stats.spearmanr(edss_data, CSA_C2_ref_mean_lst))))
print("Korelacija med CSA3 in EDSS (referenčne vrednosti CSA): "
      "Spearmanov rank: {:.4f}, p-vrednost: {:.4f}".format(*list(stats.spearmanr(edss_data, CSA_C3_ref_mean_lst))))

# prikazi raztrosni diagram odvisnosti CSA od EDSS
f, ax = plt.subplots(1, 2, sharey=True)
f.suptitle("Izračunane vrednosti CSA")
for patient_ind in range(NUMBER_OF_USED_PATIENTS):
    edss_c2_vec = np.repeat(edss_data[patient_ind], len(csa2_preds_lst[patient_ind]))
    edss_c3_vec = np.repeat(edss_data[patient_ind], len(csa3_preds_lst[patient_ind]))
    ax[0].scatter(edss_c2_vec, csa2_preds_lst[patient_ind], c='b')
    ax[0].set_title("Odvisnost EDSS od CSA2")
    ax[0].set_xlabel("EDSS")
    ax[0].set_ylabel("CSA [mm^2]")
    ax[1].scatter(edss_c3_vec, csa3_preds_lst[patient_ind], c='b')
    ax[1].set_title("Odvisnost EDSS od CSA3")
    ax[1].set_xlabel("EDSS")

# prikazi raztrosni diagram odvisnosti CSA od EDSS za referencne podatke
f, ax = plt.subplots(1, 2, sharey=True)
f.suptitle("Referenčne vrednosti CSA")
for patient_ind in range(NUMBER_OF_USED_PATIENTS):
    edss_c2_vec = np.repeat(edss_data[patient_ind], len(csa2_ref_lst[patient_ind]))
    edss_c3_vec = np.repeat(edss_data[patient_ind], len(csa3_ref_lst[patient_ind]))
    ax[0].scatter(edss_c2_vec, csa2_ref_lst[patient_ind], c='b')
    ax[0].set_title("Odvisnost EDSS od CSA2")
    ax[0].set_xlabel("EDSS")
    ax[0].set_ylabel("CSA [mm^2]")
    ax[1].scatter(edss_c3_vec, csa3_ref_lst[patient_ind], c='b')
    ax[1].set_title("Odvisnost EDSS od CSA3")
    ax[1].set_xlabel("EDSS")


# Bland-Altman analiza
f, ax = plt.subplots(2, 1, sharey=True, sharex=True)
analysisBlandAltman('CSA 2', np.concatenate(csa2_preds_lst), np.concatenate(csa2_ref_lst), iAxes=ax[0])
analysisBlandAltman('CSA 3', np.concatenate(csa3_preds_lst), np.concatenate(csa3_ref_lst), iAxes=ax[1])

plt.show()
