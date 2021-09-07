from pathlib import Path
from pathlib import PosixPath
from random import sample
import pandas as pd
import shutil
from mutagen.mp3 import MP3

#use same random seed to have identical results
SEED = 23


#Ratio of files to use from dataset. i.e.  sample_ratio = 1000 takes 1/1000th of files as dataset
SAMPLE_RATIO = 100

#max length of audio clip in seconds
MAX_AUDIO_LENGTH = 4

##Audio File Directories
#Base directory with audio files
base_dir = Path('cv-corpus-7.0-singleword')

#Drill into english
english_dir = base_dir / 'en'
english_dir.resolve()

#audio clips container
clips_dir = english_dir / 'clips'
clips_dir.resolve()

data_types = ['test','train','validated']


#return list of files to use for dataset
def ingest_tsv(tsv_file):
    read_tsv =  pd.read_csv(str(tsv_file), sep='\t',usecols=['path','sentence','up_votes','down_votes'])

    #sample at correct ratio
    return (tsv_file.stem,read_tsv.sample( n = int(read_tsv.shape[0]/SAMPLE_RATIO), random_state = SEED))

def move_files(tsv_set):
    
    path = Path(tsv_set[0])

    #clean up old files
    if path.exists() and path.is_dir():
        [x.unlink() for x in path.iterdir()]
        path.rmdir()
    
    #recreate directory
    path.mkdir()
    
    #copy files to data directories if they are shorter than max length
    for x in tsv_set[1]['path']:
        file = clips_dir / x 
        audio = MP3(file)
        if audio.info.length <= MAX_AUDIO_LENGTH:
            print(audio.info.bitrate,audio.info.length)
            shutil.copy(file, path) 

    tsv_set[1].to_csv(index=False, path_or_buf = tsv_set[0] + '/contents.csv',columns = ['path','sentence'])

def main():
    tsv_paths= [ PosixPath(english_dir,x + '.tsv') for x in data_types]
    


    tsv_files = [ingest_tsv(x) for x in tsv_paths]
    [move_files(x) for x in tsv_files]


if __name__ == "__main__":
    main()

