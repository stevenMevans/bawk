from pathlib import Path
from pathlib import PosixPath
from random import sample
import pandas as pd
import shutil

#use same random seed to have identical results
random_seed = 23


#Ratio of files to use from dataset. i.e.  sample_ratio = 1000 takes 1/1000th of files as dataset
sample_ratio = 100

##Audio File Directories
#Base directory with audio files
base_dir = Path('cv-corpus-6.1-singleword')

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
    return (tsv_file.stem,read_tsv.sample( n = int(read_tsv.shape[0]/sample_ratio), random_state = random_seed))

def move_files(tsv_set):
    
    path = Path(tsv_set[0])

    #clean up old files
    if path.exists() and path.is_dir():
        [x.unlink() for x in path.iterdir()]
        path.rmdir()
    
    #recreate directory
    path.mkdir()
    
    #copy files to data directories
    for x in tsv_set[1]['path']:
        file = clips_dir / x
        shutil.copy(file, path)

    tsv_set[1].to_csv(index=False, path_or_buf = tsv_set[0] + '/contents.csv',columns = ['path','sentence'])

def main():
    tsv_paths= [ PosixPath(english_dir,x + '.tsv') for x in data_types]
    


    tsv_files = [ingest_tsv(x) for x in tsv_paths]
    [move_files(x) for x in tsv_files]
    #print (tsv_files)

if __name__ == "__main__":
    main()

