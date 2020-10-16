#!/bin/bash

# This script prepare Callhome dataset with 5 fold cross validation
# Callhome dataset doesn't specify dev and test

. path.sh
. cmd.sh

stage=0

# The default data directories are for CLSP grid only.
# Please modify them to your own directories.

ami_dir=~/data/ami
musan_dir=~/data/ami/musan

if [ $stage -le 0 ]; then
  # Prepare ami
  for split in train development; do
    python scripts/ami/prepare_ami.py --data-dir $ami_dir --output-dir data/ami --split $split
    utils/data/get_utt2dur.sh data/ami/$split
    utils/utt2spk_to_spk2utt.pl data/ami/$split/utt2spk > data/ami/$split/spk2utt
    cp $ami_dir/AMI/MixHeadset.${split}.rttm data/ami/$split/rttm || exit 1;
    python scripts/create_spk2idx.py data/ami/$split || exit 1;
    cp data/ami/$split/utt2dur data/ami/$split/reco2dur
  done
fi
  
uttdur=10.0
if [ $stage -le 1 ]; then
  # split dataset into 10s chunks
  for split in train development; do
    scripts/split_utt.sh --cmd "$train_cmd" --nj 10 --sample_rate 16000 data/ami/$split data/ami_10s/$split || exit 1;
    awk -F' ' -v dur="$uttdur" '{print $1, dur}' data/ami_10s/$split/wav.scp > data/ami_10s/$split/reco2dur
    # data augmentation
    scripts/augmentation.sh --sample_rate 16000 --musan_dir $musan_dir data/ami_10s/$split || exit 1;
    awk -F' ' -v dur="$uttdur" '{print $1, dur}' data/ami_10s/${split}_combined/wav.scp > data/ami_10s/${split}_combined/reco2dur
    # collect information for each 10s segment for training convenience
    scripts/create_record.sh --cmd "$train_cmd" --nj 10 data/ami_10s/${split}_combined
  done
fi