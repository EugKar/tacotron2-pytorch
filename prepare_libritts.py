import csv
import re
from pathlib import Path
from argparse import ArgumentParser

def parse_tsv(tsv_path):
    ret = []
    wav_dir = tsv_path.parent
    with open(tsv_path, 'r') as fid:
        reader = csv.reader(fid, dialect='excel-tab')
        speaker_id = tsv_path.parents[1].name
        for row in reader:
            audio_path = wav_dir / row[0]
            label = row[1] if len(row) < 3 else row[2]
            ret.append((audio_path, label, speaker_id))
    return ret

def parse_set(root_dir):
    ret = []
    for speaker_dir in root_dir.iterdir():
        if not speaker_dir.is_dir():
            continue
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
            for tsv in chapter_dir.glob('*.tsv'):
                if 'trans' in tsv.name:
                    ret.extend(parse_tsv(tsv))
    return ret

def main(args):
    root_dir = Path(args.libritts_path)
    out_dir = Path(args.out_path)
    train_file = out_dir / 'libritts_audio_text_train_filelist'
    train_set = parse_set(root_dir / 'train-clean-100')
    with open(train_file, 'w') as fid:
        for item in train_set:
            fid.write(f'{item[0]}|{item[1]}|{item[2]}\n')
    val_file = out_dir / 'libritts_audio_text_val_filelist'
    val_set = parse_set(root_dir / 'dev-clean')
    with open(val_file, 'w') as fid:
        for item in val_set:
            fid.write(f'{item[0]}|{item[1]}|{item[2]}\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--libritts_path', type=str, help='LibriTTS path.')
    parser.add_argument('--out_path', type=str, help='Directory for output index.')
    args = parser.parse_args()
    main(args)