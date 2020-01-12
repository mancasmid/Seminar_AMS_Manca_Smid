from keras.models import Model, load_model
from keras.layers import Input, MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers.core import Dropout, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import random
from funkcije import *

T1_IMAGES_PATH = r'.\Gradivo\T1_slike\t1-images'
SCT_RESULTS_PATH = r'.\Gradivo\sct_results'

patient_paths = os.listdir(SCT_RESULTS_PATH)

# ustvari 'models' in 'graphs' direktorija, ce ne obstajata
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('graphs'):
    os.makedirs('graphs')

print("Zlaganje slik")
OUTPUT_SIZE = (128, 128)
NUMBER_OF_USED_PATIENTS = 28

# dobi mri podatke
t1_array, seg_array, _ = get_mri_data(SCT_RESULTS_PATH, OUTPUT_SIZE, NUMBER_OF_USED_PATIENTS, plane='transversal', extraction_index=(30, 30, 0))

# normaliziraj t1 slike
t1_array = normalize_images(t1_array)

# pri opazovanju transverzalne ravnine nas zanima samo maska hrbtenjace
SPINAL_CORD = 1
seg_array[seg_array > 0] = SPINAL_CORD

# ustvari podatke za učenje in test
if K.image_data_format() == 'channels_first':
    t1_karray = t1_array[:, np.newaxis, :, :]
    seg_karray = seg_array[:, np.newaxis, :, :]
else:
    t1_karray = t1_array[:, :, :, np.newaxis]
    seg_karray = seg_array[:, :, :, np.newaxis]

NUM_SAMPLES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = t1_karray.shape

# Konstante in parametri nevronske mreze
TEST_DATA_FRACTION = 0.33  # razmerje razdelitve podatkov na testno in ucno mnozico

seed = 42
random.seed = seed
np.random.seed = seed

X_train, X_test, y_train, y_test = train_test_split(t1_karray, seg_karray, test_size=TEST_DATA_FRACTION)

print('Velikost učne zbirke slik: {}'.format(X_train.shape))
print('Velikost testne zbirke slik: {}'.format(X_test.shape))

# RAZGRADNJA
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# vhodna plast
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

S = Lambda(lambda x: x)(inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(S)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
c3 = Dropout(0.1)(c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
c4 = Dropout(0.1)(c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
p4 = MaxPooling2D((2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
c5 = Dropout(0.1)(c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)
p5 = MaxPooling2D((2, 2))(c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1])
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
model = Model(inputs=[inputs], outputs=[outputs])

# povzetek modela
model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[dice_coef])

# Hiperparametri
BATCH_SIZE = 16
NUM_EPOCHS = 100

# Ucenje modela
# zaženi učenje modela
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(os.path.join('models', 'model-mri-unet-seg-transversal.h5'), verbose=1, save_best_only=True)
results = model.fit(X_train, y_train, validation_split=0.1, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                    callbacks=[earlystopper, checkpointer])

# Nalozi model in izracunaj razgradnje na ucni in testni zbirki
model = load_model(os.path.join('models', 'model-mri-unet-seg-transversal.h5'),
                   custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})

# opravi razgradnjo na učni in testni zbirki
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# dobljene vrednosti
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

MODALITIES = ['T1']

# preveri kakovost razgradnje na učnih vzorcih (sanity check)
ix = random.randint(0, len(preds_train_t)-1)
_, _, num_modalities = X_train[ix].shape

titles = [m.upper() + ' slika' for m in MODALITIES] + ['Referenčna razgradnja', 'Razgradnja U-net']
f, ax = plt.subplots(1, num_modalities+2, sharex=True, sharey=True, figsize=(20, 5))
for i in range(num_modalities):
    ax[i].imshow(X_train[ix][:, :, i], cmap='gray')
    ax[i].set_title(titles[i])
    ax[i].axis('off')
# prikaži referenčno razgradnjo
ax[-2].imshow(np.squeeze(y_train[ix]))
ax[-2].set_title(titles[-2])
ax[-2].axis('off')
# prikaži razgradnjo z U-net
ax[-1].imshow(np.squeeze(preds_train_t[ix]))
ax[-1].set_title(titles[-1])
ax[-1].axis('off')

# Kvalitativno: kakovost razgradnje na testnih vzorcih (zmožnost posplosevanja)
# preveri kakovost razgradnje na naključno izbranih testnih vzorcih
ix = random.randint(0, len(preds_test_t)-1)
_, _, num_modalities = X_test[ix].shape

titles = [m.upper() + ' slika' for m in MODALITIES] + ['Referenčna razgradnja', 'Razgradnja U-net']
f, ax = plt.subplots(1, num_modalities+2, sharex=True, sharey=True, figsize=(20, 5))
for i in range(num_modalities):
    ax[i].imshow(X_test[ix][:, :, i], cmap='gray')
    ax[i].set_title(titles[i])
    ax[i].axis('off')
# prikaži referenčno razgradnjo
ax[-2].imshow(np.squeeze(y_test[ix]), cmap='gray')
ax[-2].set_title(titles[-2])
ax[-2].axis('off')
# prikaži razgradnjo z U-net
ax[-1].imshow(np.squeeze(preds_test_t[ix]), cmap='gray')
ax[-1].set_title(titles[-1])
ax[-1].axis('off')


# Kvalitativno: povprecna vrednost Diceovega koeficienta na vseh testnih slikah
def dice(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    intersection = np.count_nonzero(result & reference)
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    return dc


train_dice, test_dice = [], []
for i in range(y_test.shape[0]):
    train_dice.append(dice(preds_train_t[i].flatten(), y_train[i].flatten()))
    test_dice.append(dice(preds_test_t[i].flatten(), y_test[i].flatten()))

print('Povprečna vrednost Diceovega koeficienta na učni zbirki: ', np.mean(train_dice))
print('Povprečna vrednost Diceovega koeficienta na testni zbirki: ', np.mean(test_dice))

plt.show()
