import csv
import re
import random
from pathlib import Path
from argparse import ArgumentParser
from itertools import chain

def parse_tsv(tsv_path, wav_dir):
    ret = []
    speakers = set()
    with open(tsv_path, 'r') as fid:
        reader = csv.reader(fid, dialect='excel-tab')
        for row in reader:
            audio_path = wav_dir / f'{row[0]}.wav'
            speaker_id = re.search(r'^\w+_(\d+)_\d+$', row[0])[0]
            ret.append((audio_path, row[1], speaker_id))
            speakers.add(speaker_id)
    return ret, speakers
            

def main(args):
    random.seed(42)
    tsv_dir = Path(args.tsv_dir)
    wav_dir = Path(args.wav_dir)
    items_female, female_speakers = parse_tsv(tsv_dir / 'line_index_female.tsv', wav_dir)
    items_male, male_speakers = parse_tsv(tsv_dir / 'line_index_male.tsv', wav_dir)
    assert len(male_speakers & female_speakers) == 0, 'Male and female spakers ids intersect!'
    items_count = len(items_female) + len(items_male)
    test_val_inds = random.sample(range(items_count), args.val_count + args.test_count)
    test_inds = random.sample(test_val_inds, args.test_count)

    out_path = Path(args.out_path)
    test_file = out_path / 'spanish_audio_text_test_filelist.txt'
    val_file = out_path / 'spanish_audio_text_val_filelist.txt'
    train_file = out_path / 'spanish_audio_text_train_filelist.txt'

    with open(test_file, 'w') as f_test, open(train_file, 'w') as f_train, open(val_file, 'w') as f_val:
        for i, item in enumerate(chain(items_female, items_male)):
            if i in test_inds:
                fid = f_test
            elif i in test_val_inds:
                fid = f_val
            else:
                fid = f_train
            fid.write(f'{item[0]}|{item[1]}|{item[2]}\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tsv_dir', type=str, help='Directory with utterances index.')
    parser.add_argument('--wav_dir', type=str, help='Directory with audio files.')
    parser.add_argument('--out_path', type=str, help='Directory for output index.')
    parser.add_argument('--val_count', type=int, default=50, help='Number of validation recordings.')
    parser.add_argument('--test_count', type=int, default=100, help='Number of test recordings')
    args = parser.parse_args()
    main(args)