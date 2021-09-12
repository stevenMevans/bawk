from seqtoseq import *
from  create_dataset import preprocess,train_text,train_path
from train import trainIters
from predict import evaluateRandomly, inference_from_file
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformed_dataset = preprocess(train_manifest_path='/Users/dami.osoba/work/bawk/small_dataset/commonvoice_train_manifest.json')


enc_path = 'output/enc_model'
encoder = torch.load(enc_path)
encoder.eval()

dec_path = 'output/dec_model'
decoder = torch.load(enc_path)
decoder.eval()

trainIters(transformed_dataset,encoder, decoder, 10000, print_every=1000, plot_every=1000)
evaluateRandomly(transformed_dataset,encoder, encoder,10)

# save models
enc_path = "output/enc_model"
torch.save(encoder.state_dict(), enc_path)
dec_path = "output/dec_model"
torch.save(decoder.state_dict(), dec_path)

enc_path = "enc_model_spare"
torch.save(encoder, enc_path)
dec_path = "dec_model_spare"
torch.save(decoder, dec_path)

