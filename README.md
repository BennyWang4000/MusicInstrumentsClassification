# MusicInstrumentsClassification
## Usage
```python
from instruments.model import TransformerClassifier

model= TransformerClassifier(20)
model.load('state_dict_path.pt')
result= model()
```

## Dataset
OpenMIC-2018
- inexact label
- multi-label

## Method
- PyTorch pre-trained VGGish
- transformer encoder
- three layer fully connect with dropout and leakyReLU

## Result
:P