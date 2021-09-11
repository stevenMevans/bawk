from seqtoseq import *
from create_dataset import preprocess
from train import trainIters
from predict import evaluateRandomly
import torch
import pickle
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs= 100000
print_evy = 1000
model_path = "model_simple_small"

transformed_dataset = preprocess(train_manifest_path='/Users/dami.osoba/work/cv-corpus-6.1-2020-12-11/en/commonvoice_fourth_small_manifest.json')

encoder1 = EncoderRNN(mels_dims*MAX_LENGTH, 256).to(device)

# encoder1 = EncoderRNN(mels_dims, 256).to(device)
decoder1 = AttnDecoderRNN(256, 29, dropout_p=0.1).to(device)



# encoder1 = Encoder(80, 40, 2, dropout=0.05, bidirectional=True).to(device)
# decoder1 = Decoder(vocab_size=29, embedding_dim=15, hidden_size=20, num_layers=2).to(device)



trainIters(transformed_dataset,encoder1, decoder1, n_iters=epochs,  model_save_path=model_path, print_every=print_evy,
           learning_rate=0.001, reload_path=None)

evaluateRandomly(transformed_dataset,encoder1, decoder1,10)

# reload(transformed_dataset,encoder1,decoder1, 500, print_every=100, learning_rate=0.0001)



# file
# with open('output/encoder_vars_simple.pkl', 'wb') as convert_file:
#     pickle.dump(encoder1,convert_file)
#
# with open('output/decoder_vars_simple.pkl', 'wb') as convert_file:
#     pickle.dump(decoder1,convert_file)
#
#
enc_path = f"output/enc_{model_path}"
torch.save(encoder1.state_dict(), enc_path)
dec_path = f"output/dec_{model_path}"
torch.save(decoder1.state_dict(), dec_path)







