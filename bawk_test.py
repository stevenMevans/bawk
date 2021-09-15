from models.rename_unpickler import renamed_load
import torch
from models.inference import inference_from_file
from src.api.keyword_detection_service import KeywordDetectionService
from nltk.util import ngrams
import pickle
import pandas as pd
import csv

def run_inference(filepath,keyword):
    sentence = inference_from_file(filepath, encoder, decoder,False)

    detector = KeywordDetectionService(keyword)
    ret_msg = detector.check_text(sentence)
    return(ret_msg)

def analyze_ngrams(file,ngram_length = 3,top_ngrams = 100):
    counts = {}

    read_tsv =  pd.read_csv(str(file), sep='\t',usecols=['sentence','path'])

    for index,row in read_tsv.iterrows():
        sentence = row['sentence']
        path = row['path']
        
        if not pd.isna(sentence):
            for ngram in set(ngrams(KeywordDetectionService.tokenize(sentence),ngram_length)):
                if ngram not in counts.keys():
                    counts[ngram] = { "count" : 0, "paths" : []}

                counts[ngram]['paths'].append(path)
                counts[ngram]['count'] += 1

    a_sorted = sorted(counts.items(), key = lambda kv:kv[1]['count'],reverse=True)[:top_ngrams]
    return [(' '.join(x[0]),x[1]['paths']) for x in a_sorted]

def output_results(results,filename = 'keyword_detection.csv'):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['phrase', 'phrase_length','True','False']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for kv in results.items():
            writer.writerow({'phrase': kv[0], 'phrase_length': len(kv[0].split()),'True' : kv[1][True],'False' : kv[1][False]})
        


# initial setup
model_path = 'models/model_las_updated_final.pth'
encoder_pkl_path = 'models/encoder_las.pkl'
decoder_pkl_path = 'models/decoder_las.pkl'

device = torch.device("cpu")

with open(encoder_pkl_path, 'rb') as convert_file:
    encoder = renamed_load(convert_file)

with open(decoder_pkl_path, 'rb') as convert_file:
    decoder = renamed_load(convert_file)

checkpoint = torch.load(model_path, map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
encoder.eval()
decoder.load_state_dict(checkpoint['decoder_state_dict'])
decoder.eval()
# end setup


# find the most common phrases in the test set
keywords = analyze_ngrams("src/data/cv-corpus-6.1-2020-12-11/en/test.tsv")

results = {}
for blob in keywords:
    phrase  = blob[0]
    for path in blob[1]:  
        res = run_inference("src/data/cv-corpus-6.1-2020-12-11/en/clips/"+ path,phrase) 
        if phrase not in results.keys():
            results[phrase] = {}
        if res not in results[phrase].keys():
            results[phrase][res] = 0
        results[phrase][res] += 1
print(results)
output_results(results)
