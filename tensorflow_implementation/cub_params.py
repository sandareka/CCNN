# -------- Dataset information -------
TRAIN_IMAGES_DIR = 'data/train'
TEST_IMAGES_DIR = 'data/test'
INDICATOR_VECTORS = 'data/image_indicator_vectors.npy'
CLASS_DISCRIMINATIVE_FEATURE = 'data/class_discriminative_features.npy'
PRE_TRAINED_MODEL_WEIGHTS = 'pre_trained_models/vgg_16.ckpt'
CONCEPT_WORD_VECTORS = 'data/concept_word_vectors.npy'

NO_CLASSES = 200
CLASS_NAME_FILE = "data/classes.txt"

# -------- CCNN parameters --------
LAMBDA_VALUE = 0.4
BETA = 0.5
ALPHA = 1
EMBED_SPACE_DIM = 24
TEXT_SPACE_DIM = 50
NO_CONCEPTS = 398

# -------- Training parameters --------
MODEL_SAVE_FOLDER = "ccnn_trained_model/"
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
NUM_WORKERS = 3
MAX_NUM_EPOCHS_1 = 20
MAX_NUM_EPOCHS_2 = 50
MAX_NUM_EPOCHS_3 = 20
LEARNING_RATE_1 = 1e-3
LEARNING_RATE_2 = 1e-4
LEARNING_RATE_3 = 1e-4
DROP_OUT_KEEP_PROB = 0.5
WEIGHT_DECAY = 5e-4
VGG_MEAN = [123.68, 116.78, 103.94]
