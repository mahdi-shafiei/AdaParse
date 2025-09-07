# torch on aurora
import torch
if torch.xpu.is_available():
    import intel_extension_for_pytorch as ipex
#from transformers import VisionEncoderDecoderModel, NougatProcessor

if __name__=='__main__':
    print('\n\n\nSuccess')

    pass
