from torch import device, cuda, optim
import torch.nn

# * ============== data ==============

MODEL_NAME = 'transformer'

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

INST2CLASSIDX_DICT = {
    'drums': 0,
    'cymbals': 0,
    'saxophone': 1,
    'trombone': 1,
    'trumpet': 1,
}

CLASSIDX2NAME_DICT = {
    0: 'drums',
    1: 'brass'
}

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

# * ============== training ==============

TEST_PER = 0.1
IS_VALID = True
K_FOLDS = 10

EPOCHS = 5
BATCH_SIZE = 4
LR = 0.0001

OPTIMIZER = optim.Adam
BETA1 = 0.75
BETA2 = 2.0
CRITERION = torch.nn.CrossEntropyLoss()
# CRITERION = torch.nn.()

IS_WANDB = True


# * ============== signal ==============

AUDIO_LENGTH = 441000
SAMPLE_RATE = 44100
HOP_LENGTH = 441
N_FFT = 2048
N_MELS = 64
F_MIN = 0
F_MAX = 22050
N_FREQS = 1024
WIN_LENGTH = 128


# * ============== torch ==============

IS_CUDA = cuda.is_available()
DEVICE = device('cuda:0' if cuda.is_available() else 'cpu')

PASST_PATH = './pre_trained/passt-l-kd-ap.47.pt'
VGGISH_PATH = './pre_trained/vggish-10086976.pth'
