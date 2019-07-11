import csv
import re
from pathlib import Path
from argparse import ArgumentParser

def parse_tsv(tsv_path, speakers_vocab=[]):
    ret = []
    wav_dir = tsv_path.parent
    with open(tsv_path, 'r') as fid:
        reader = csv.reader(fid, dialect='excel-tab')
        speaker_id = tsv_path.parents[1].name
        if speaker_id not in speakers_vocab:
            speakers_vocab.append(speaker_id)
        speaker_id = speakers_vocab.index(speaker_id)
        for row in reader:
            audio_path = wav_dir / (row[0] + '.wav')
            label = row[1] if len(row) < 3 else row[2]
            ret.append((audio_path, label, speaker_id))
    return ret

def parse_set(root_dir, speakers_vocab=[]):
    ret = []
    for speaker_dir in root_dir.iterdir():
        if not speaker_dir.is_dir():
            continue
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
            for tsv in chapter_dir.glob('*.tsv'):
                if 'trans' in tsv.name:
                    ret.extend(parse_tsv(tsv, speakers_vocab))
    return ret

def main(args):
    root_dir = Path(args.libritts_path)
    out_dir = Path(args.out_path)
    speakers_vocab = []

    train_file = out_dir / 'libritts_audio_text_train_filelist.txt'
    train_set = parse_set(root_dir / 'train-clean-100', speakers_vocab)
    with open(train_file, 'w') as fid:
        for item in train_set:
            fid.write(f'{item[0]}|{item[1]}|{item[2]}\n')
    val_file = out_dir / 'libritts_audio_text_val_filelist.txt'
    val_set = parse_set(root_dir / 'dev-clean', speakers_vocab)
    dev_size = int(args.dev_size)
    if dev_size > 0:
        val_set = val_set[:dev_size]
    with open(val_file, 'w') as fid:
        for item in val_set:
            fid.write(f'{item[0]}|{item[1]}|{item[2]}\n')
    speakers_vocab_file = out_dir / 'speakers.txt'
    with open(speakers_vocab_file, 'w') as fid:
        fid.write('\n'.join(speakers_vocab))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--libritts_path', type=str, help='LibriTTS path.')
    parser.add_argument('--out_path', type=str, help='Directory for output index.')
    parser.add_argument('--dev_size', type=int, default=-1, help='Dev set size, -1 for the whole dev set')
    args = parser.parse_args()
    main(args)