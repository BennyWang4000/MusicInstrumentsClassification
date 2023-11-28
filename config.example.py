from torch import device, cuda, optim
import torch.nn.functional as F
# * ============== data ==============

OPENMIC_DIR = './dataset/openmic-2018'
MODEL_STATE_DIR = './runs'
CLASS2IDX_DICT = {"accordion": 0, "banjo": 1, "bass": 2, "cello": 3, "clarinet": 4, "cymbals": 5, "drums": 6, "flute": 7, "guitar": 8, "mallet_percussion": 9,
                  "mandolin": 10, "organ": 11, "piano": 12, "saxophone": 13, "synthesizer": 14, "trombone": 15, "trumpet": 16, "ukulele": 17, "violin": 18, "voice": 19}
IDX2CLASS_DICT = {v: k for k, v in CLASS2IDX_DICT.items()}

# * ============== training ==============

TEST_PER = 0.2
VALID_PER = 0.1
IS_VALID = True

EPOCHS = 10
BATCH_SIZE = 8
LR = 0.0002

OPTIMIZER = optim.Adam
CRITERION = F.cross_entropy

IS_WANDB = True

# * ============== signal ==============

SAMPLE_RATE = 16000


# * ============== torch ==============

IS_CUDA = cuda.is_available()
DEVICE = device("cuda:0" if cuda.is_available() else "cpu")
