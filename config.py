import string

# learning_rate (tmbh 0), batchsize kelipatan 2, epochs
ALPHABET = string.ascii_lowercase
NUMBER = string.digits
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 30
IMG_H = 32
IMG_W = 100
NUM_TRAIN = 176000  # total - 20% 176000
NUM_VAL = 44000  # total - 80% 44000
OUT_DIR = "modelCRNN/save_model"
OUT_DIR_HISTORY = "modelCRNN/save_history/"

# setingan epoch 30
# LEARNING_RATE = 0.0001
# BATCH_SIZE = 16

# EPOCHS = 20
# LEARNING_RATE = 0.0005
# BATCH_SIZE = 32

# EPOCH = 40
# LEARNING_RATE = 0.00005
# BATCH_SIZE = 64