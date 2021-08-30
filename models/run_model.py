from seqtoseq import *
from models.create_dataset import preprocess, train_text, train_path
from train import trainIters
from models.predict import evaluateRandomly, inference_from_file
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformed_dataset = preprocess(train_manifest_path='/Users/dami.osoba/work/bawk/small_dataset/commonvoice_train_manifest.json')

encoder1 = EncoderRNN(mels_dims*MAX_LENGTH, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, 29, dropout_p=0.1).to(device)

trainIters(transformed_dataset,encoder1, attn_decoder1, 10000, print_every=1000, plot_every=1000)
evaluateRandomly(transformed_dataset,encoder1, attn_decoder1,10)

# save models
enc_path = "enc_model"
torch.save(encoder1.state_dict(), enc_path)
dec_path = "dec_model"
torch.save(attn_decoder1.state_dict(), dec_path)

enc_path = "enc_model_spare"
torch.save(encoder1, enc_path)
dec_path = "dec_model_spare"
torch.save(attn_decoder1, dec_path)

