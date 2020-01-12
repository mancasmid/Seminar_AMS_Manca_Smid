# AMS_seminar
Seminar: Lokalizacija hrbtenjače v MR slikah  
Avtorica: Manca Šmid  

**Vsebina**:  
*mri_unet_transverzalna_ravnina.py*: določitev U-Net modela za lokalizacijo hrbtenjače  
*mri_unet_sagitalna_ravnina.py*: določitev U-Net modela za razgradnjo hrbtenjače na C2 in C3  
*izracun_csa.py*: Izračun CSA C2/C3 za vsakega pacienta  
*validacija_modela.py*: Validacija modelov z izračunom korelacije z EDSS  
*funkcije.py*: Različne funkcije, ki jih uporabljajo skripte  
*amslib.py*: AMS knjižnica  
*requirements.txt*: seznam potrebnih Python knjižnic  
*Gradivo*: direktorij z gradivom. Ker vsebuje večje datoteke, je potrebno direktorij ročno prenesti iz Google Drive (glej navodila).

**Navodila za uporabo**:
1. Iz Google drive (https://drive.google.com/open?id=1G9kqVEkeTAuqOWvrrqRmmLaNvD1tSO3t) prenesite direktorij Gradivo in ga premaknite v projekt
2. Namen tega koraka je pridobiti modela U-Net v obliki .h5 datotek za kasnejšo uporabo.
   Primeri že dobljenih datotek se že nahajajo v direktoriju *models*, zato lahko ta korak preskočite. 
   1. Poženite mri_unet_transverzalna_ravnina.py in mri_unet_sagitalna_ravnina.py, ki bosta ustvarila
        U-Net modela, ki se bosta shranila v datoteki *./models/model-mri-unet-seg-transversal.h5*
        ter *./models/model-mri-unet-seg-sagital.h5*. V skriptah lahko prirejate hiperparametre postopka učenja.  
   2. Ko ste zadovoljni z dobljenim modelom, preimenujte omenjeni .h5 datoteki v
        *model-mri-unet-seg-sagital-best.h5* ter *model-mri-unet-seg-transversal-best.h5*.
3. Poženite skripto izracun_csa.py. Skripta bo na podlagi najboljših najdenih U-Net modelov
   iz vhodnih T1 MRI slik opravila razgradnjo hrbtenjače ter izračunala CSA C2/C3
   za vsakega pacienta. Rezultati bodo shranjeni v datoteko *csa_data.npz*. Primer te datoteke
   je že pripravljen vnaprej, zato lahko ta korak preskočite.
4. Poženite skripto *validacija_modela.py*. Skripta bo naložila izračunane CSA vrednosti iz *csa_data.npz* ter podatke EDSS
   pacientov. Iz teh podatkov se bo izračunala korelacija med EDSS in izračunanimi CSA vrednostmi.
   Rezultati bodo izpisani v konzoli in prikazani grafično.
   
 **Programsko okolje**  
Python 3.6.5  
Uporabljene knjižnice: glej requirements.txt
