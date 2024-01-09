from torch import device, cuda, optim
import torch.nn

# * ---------------------------------------------------------------------------- #
# *                                     data                                     #
# * ---------------------------------------------------------------------------- #
MODEL_NAME = 'flatten outside'
OPENMIC_DIR = './data/openmic-2018'
MODEL_STATE_DIR = './runs'

INST2IDX_DICT = {'accordion': 0,
                 'banjo': 1,
                 'bass': 2,
                 'cello': 3,
                 'clarinet': 4,
                 'cymbals': 5,
                 'drums': 6,
                 'flute': 7,
                 'guitar': 8,
                 'mallet_percussion': 9,
                 'mandolin': 10,
                 'organ': 11,
                 'piano': 12,
                 'saxophone': 13,
                 'synthesizer': 14,
                 'trombone': 15,
                 'trumpet': 16,
                 'ukulele': 17,
                 'violin': 18,
                 'voice': 19}

IDX2INST_DICT = {v: k for k, v in INST2IDX_DICT.items()}

INST_NAME_LST = [k for k, _ in INST2IDX_DICT.items()]

# * ---------------------------------------------------------------------------- #
# *                                   training                                   #
# * ---------------------------------------------------------------------------- #
TEST_PER = 0.1
IS_VALID = False
K_FOLDS = 10

EPOCHS = 20
BATCH_SIZE = 8
LR = 0.0001

OPTIMIZER = optim.Adam
BETA1 = 0.9
BETA2 = 0.999
CRITERION = torch.nn.CrossEntropyLoss()
SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau

THRESHOLD = 0.55

SCHEDULER_MODE = 'min'
SCHEDULER_STEP_SIZE = 1000
SCHEDULER_FACTORY = 0.9

IS_WANDB = False


# * ---------------------------------------------------------------------------- #
# *                                  pre-trained                                 #
# * ---------------------------------------------------------------------------- #

PRE_TRAINED = 'vggish'
IS_PRE_TRAINED_EVAL = False

PRE_TRAINED_PATHS = {
    'passt': './pre_trained/passt-l-kd-ap.47.pt',
    'vggish': './pre_trained/vggish-10086976.pth'
}

# * ---------------------------------------------------------------------------- #
# *                                    signal                                    #
# * ---------------------------------------------------------------------------- #

AUDIO_LENGTH = 160000
SAMPLE_RATE = 16000
HOP_LENGTH = 441
N_FFT = 2048
N_MELS = 64
F_MIN = 0
F_MAX = 22050
N_FREQS = 1024
WIN_LENGTH = 128

KWARGS_SIGNAL = {
    'audio_length': AUDIO_LENGTH,
    'sample_rate': SAMPLE_RATE,
    'hop_length': HOP_LENGTH,
    'n_fft': N_FFT,
    'n_mels': N_MELS,
    'f_min': F_MIN,
    'f_max': F_MAX,
    'n_freqs': N_FREQS,
    'win_length': WIN_LENGTH,
}


# * ---------------------------------------------------------------------------- #
# *                                     torch                                    #
# * ---------------------------------------------------------------------------- #

IS_CUDA = cuda.is_available()
DEVICE = device('cuda:0' if cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------- #
#                                  deprecated                                  #
# ---------------------------------------------------------------------------- #
# INST2CLASSIDX_DICT = {'bass': 0,
#                       'drums': 1,
#                       'guitar': 2,
#                       'piano': 3,
#                       'trombone': 4,
#                       'trumpet': 4,
#                       }
# INSTIDX2CLASSIDX_DICT = {2: 0,
#                          6: 1,
#                          8: 2,
#                          12: 3,
#                          #  15: 4,
#                          #  16: 4,
#                          19: 5,
#                          }
# INSTIDX2CLASSIDX_DICT = {15: 0,
#                          16: 0,
#                          0: 1,
#                          4: 1,
#                          7: 1,
#                          11: 1,
#                          13: 1,
#                          1: 2,
#                          2: 2,
#                          3: 2,
#                          8: 2,
#                          10: 2,
#                          17: 2,
#                          18: 2,
#                          5: 3,
#                          6: 3,
#                          14: 4,
#                          9: 5,
#                          12: 5,
#                          19: 6
#                          }
# INST2CLASSIDX_DICT = {
#     k: INSTIDX2CLASSIDX_DICT.get(v, 99) for k, v in INST2IDX_DICT.items()}
# CLASS2IDX_DICT = {0: 'brass',
#                   1: 'wood-wing',
#                   2: 'string',
#                   3: 'drums',
#                   4: 'synthesizer',
#                   5: 'piano',
#                   6: 'vocal', }
# INST2CLASSIDX_DICT = {
#     'drums': 0,
#     'cymbals': 0,
#     'saxophone': 1,
#     'trombone': 1,
#     'trumpet': 1,
# }

# CLASSIDX2NAME_DICT = {
#     0: 'drums',
#     1: 'brass'
# }
