from seqtoseq import *
from create_dataset import preprocess
from train import trainIters
from predict import evaluateRandomly
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformed_dataset = preprocess(train_manifest_path='/Users/dami.osoba/work/bawk/small_dataset/commonvoice_train_manifest.json', cloud=True)

encoder1 = EncoderRNN(mels_dims*MAX_LENGTH, 256).to(device)
decoder1 = AttnDecoderRNN(256, 29, dropout_p=0.1).to(device)


# encoder1 = Encoder(80, 40, 2, dropout=0.05, bidirectional=True).to(device)
# decoder1 = Decoder(vocab_size=29, embedding_dim=15, hidden_size=20, num_layers=2).to(device)

trainIters(transformed_dataset,encoder1, decoder1, 10000, print_every= 100,learning_rate=0.01)
evaluateRandomly(transformed_dataset,encoder1, decoder1,10)



