from seqtoseq import *
from  create_dataset import preprocess
from train import trainIters
from predict import evaluateRandomly
import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformed_dataset = preprocess(train_manifest_path='/Users/dami.osoba/work/bawk/small_dataset/commonvoice_train_manifest.json')

encoder1 = EncoderRNN(80, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, 29, dropout_p=0.1).to(device)

trainIters(transformed_dataset,encoder1, attn_decoder1, 2000, print_every=1000, plot_every=1000)
evaluateRandomly(transformed_dataset,encoder1, attn_decoder1)
