from seqtoseq import *
from  create_dataset import preprocess,train_text,train_path
from train import trainIters
from predict import evaluateRandomly, inference_from_file
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformed_dataset = preprocess(train_manifest_path='/Users/dami.osoba/work/bawk/small_dataset/commonvoice_train_manifest.json')

encoder1 = EncoderRNN(mels_dims*MAX_LENGTH, hidden_size).to(device)
decoder1 = AttnDecoderRNN(hidden_size, 29, dropout_p=0.1).to(device)


# encoder1 = Encoder(80, 10, 2, dropout=0.05, bidirectional=True).to(device)
# decoder1 = Decoder(vocab_size=29, embedding_dim=15, hidden_size=20, num_layers=1).to(device)

trainIters(transformed_dataset,encoder1, decoder1, 1000, print_every= 100,learning_rate=0.01)
evaluateRandomly(transformed_dataset,encoder1, decoder1,10)

# # save models
# enc_path = "enc_model_las"
# torch.save(encoder1.state_dict(), enc_path)
# dec_path = "dec_model_las"
# torch.save(decoder1.state_dict(), dec_path)


