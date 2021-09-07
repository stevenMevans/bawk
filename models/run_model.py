from seqtoseq import *
from create_dataset import preprocess
from train import trainIters, reload
from predict import evaluateRandomly
import torch
import pickle
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformed_dataset = preprocess(train_manifest_path='/Users/dami.osoba/work/bawk/small_dataset/commonvoice_train_manifest.json', cloud=True)

encoder1 = EncoderRNN(mels_dims*MAX_LENGTH, 256).to(device)

# encoder1 = EncoderRNN(mels_dims, 256).to(device)
decoder1 = AttnDecoderRNN(256, 29, dropout_p=0.1).to(device)



# encoder1 = Encoder(80, 40, 2, dropout=0.05, bidirectional=True).to(device)
# decoder1 = Decoder(vocab_size=29, embedding_dim=15, hidden_size=20, num_layers=2).to(device)

trainIters(transformed_dataset,encoder1, decoder1, 10, print_every= 1,learning_rate=0.001,model_save="model_simple")
evaluateRandomly(transformed_dataset,encoder1, decoder1,10)

# reload(transformed_dataset,encoder1,decoder1, 500, print_every=100, learning_rate=0.0001)



# file
with open('output/encoder_vars_simple.pkl', 'wb') as convert_file:
    pickle.dump(encoder1,convert_file)

with open('output/decoder_vars_simple.pkl', 'wb') as convert_file:
    pickle.dump(decoder1,convert_file)


enc_path = "output/enc_model_simple"
torch.save(encoder1.state_dict(), enc_path)
dec_path = "output/dec_model_simple"
torch.save(decoder1.state_dict(), dec_path)







