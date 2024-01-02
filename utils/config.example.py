from torch import device, cuda, optim
import torch.nn

# * ---------------------------------------------------------------------------- #
# *                                     data                                     #
# * ---------------------------------------------------------------------------- #
MODEL_NAME = 'transformer+ FC scheduler .ver'
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


# * ---------------------------------------------------------------------------- #
# *                                   training                                   #
# * ---------------------------------------------------------------------------- #
TEST_PER = 0.1
IS_VALID = False
K_FOLDS = 10

EPOCHS = 10
BATCH_SIZE = 8
LR = 0.0001

OPTIMIZER = optim.Adam
BETA1 = 0.9
BETA2 = 0.999
CRITERION = torch.nn.CrossEntropyLoss()
SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau

SCHEDULER_MODE = 'min'
SCHEDULER_STEP_SIZE = 1000
SCHEDULER_FACTORY = 0.9


IS_WANDB = True


# * ---------------------------------------------------------------------------- #
# *                                  pre-trained                                 #
# * ---------------------------------------------------------------------------- #

IS_PRE_TRAINED = True
PRE_TRAINED = 'vggish'

PRE_TRAINED_PATHS = {
    'passt': './pre_trained/passt-l-kd-ap.47.pt',
    'vggish': './pre_trained/vggish-10086976.pt'
}

KWARGS_PRETRAINED = {
    'pre_trained': PRE_TRAINED,
    'PRE_TRAINED_PATH': PRE_TRAINED_PATHS[PRE_TRAINED],
}


# * ---------------------------------------------------------------------------- #
# *                                    signal                                    #
# * ---------------------------------------------------------------------------- #

AUDIO_LENGTH = 441000
SAMPLE_RATE = 44100
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
