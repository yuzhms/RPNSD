import os
import argparse
import soundfile as sf
from tqdm import tqdm

import kaldi_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/callhome_10s_combined_5folds')
    parser.add_argument('--output-dir', default='data/callhome_400ms')
    parser.add_argument('--dataset', default='callhome', choices=['callhome', 'ami'])
    args = parser.parse_args()
    return args


def process_label_file(label_file, seg_length=0.4):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    label_list = []
    spk_list = []

    # step 1: read original list
    for line in lines:
        data = line.strip().split()
        # format SPEAKER {uri} 1 {start} {duration} <NA> <NA> {identifier} <NA> <NA>
        # NOTE: here we increase spk id by 1, then use id 0 as non-speaker id
        start, duration, spk_id = float(data[3]), float(data[4]), int(data[7]) + 1
        label_list.append([start, start+duration, spk_id])
        spk_list.append(spk_id)
    label_list.sort(key=lambda _: _[0])
    spk_list = list(set(spk_list))

    # step 2:  merge conjunct utterances with the same speaker
    process_label_list = []
    for spk in spk_list:
        label_list_spk = [_ for _ in label_list if _[2] == spk]
        new_label_list_spk = []
        for i in range(len(label_list_spk)):
            if i == 0 or label_list_spk[i][0] > new_label_list_spk[-1][1]:
                # no overlap, add as another utt
                new_label_list_spk.append(label_list_spk[i])
            else:
                # has overlap, merge
                new_label_list_spk[-1][1] = label_list_spk[i][1]
        process_label_list.extend(new_label_list_spk)
    label_list = process_label_list
    label_list.sort(key=lambda _: _[0])

    # step 3: remove overlap and add zero label to non-speech (label=0)
    process_label_list = []
    occupied_region = 0

    for i, label in enumerate(label_list):
        if i == 0:
            process_label_list.append([0, label[0], 0])
            process_label_list.append(label)
            occupied_region = label[1]
        else:
            if label[0] > occupied_region:
                # no overlap
                process_label_list.append([occupied_region, label[0], 0])
                process_label_list.append(label)
                occupied_region = label[1]
            else:
                # detect overlap
                # 1. modify processed label
                process_label_list = [[_0, min(_1, label[0]), _2] for _0, _1, _2 in process_label_list]
                # 2. add new label
                if occupied_region < label[1]:
                    process_label_list.append([occupied_region, label[1], label[2]])
                    occupied_region = label[1]
    label_list = process_label_list

    # step 4: create 400ms segment list
    process_label_list = []
    for label in label_list:
        start = label[0]
        end = label[1]
        num_seg = int((end - start) / seg_length)
        for i in range(num_seg):
            process_label_list.append([start+i*seg_length, label[2]])
    label_list = process_label_list

    return label_list


def process_dataset(input_dir, output_dir):
    # load utt list
    with open(os.path.join(input_dir, 'wav.scp'), 'r') as f:
        utt_list = [_.strip().split()[0] for _ in f.readlines()]
    # prepare output dir
    os.makedirs(os.path.join(output_dir, 'wav'), exist_ok=True)

    output_label_list = []
    print('create 400ms segment...')
    for utt_name in tqdm(utt_list):
        info_file = os.path.join(input_dir, 'data', f'{utt_name}.txt')
        with open(info_file, 'r') as f:
            info = f.readline().strip()
        _, _, label_file, wav_file = info.split(None, 3)
        data, sr = kaldi_data.load_wav(wav_file)
        label_list = process_label_file(label_file)

        for idx, label in enumerate(label_list):
            start_ms = int(label[0]*1000)
            start_sample = int(label[0]*sr)
            end_ms = start_ms + 400
            end_sample = start_sample + sr * 400 // 1000
            out_filename = f'{utt_name}_L{idx}_{start_ms}_{end_ms}.wav'
            output_label_list.append(f'{label[1]} {out_filename}\n')
            sf.write(os.path.join(output_dir, 'wav', out_filename), data[start_sample: end_sample], sr)
    print('write label file...')
    with open(os.path.join(output_dir, 'label.txt'), 'w') as f:
        f.writelines(output_label_list)


if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'callhome':
        for folder_num in range(1, 6):
            callhome_10s_dir_train = os.path.join(args.data_dir, f'{folder_num}', 'train_dev_train')
            callhome_10s_dir_dev = os.path.join(args.data_dir, f'{folder_num}', 'train_dev_dev')

            callhome_400ms_dir_train = os.path.join(args.output_dir, f'{folder_num}', 'train')
            callhome_400ms_dir_dev = os.path.join(args.output_dir, f'{folder_num}', 'dev')

            print(f"processing #{folder_num} fold")
            process_dataset(callhome_10s_dir_train, callhome_400ms_dir_train)
            process_dataset(callhome_10s_dir_dev, callhome_400ms_dir_dev)
    elif args.dataset == 'ami':
        ami_10s_dir_train = os.path.join(args.data_dir, 'train_combined')
        ami_10s_dir_dev = os.path.join(args.data_dir, 'development_combined')

        ami_400ms_dir_train = os.path.join(args.output_dir, 'train')
        ami_400ms_dir_dev = os.path.join(args.output_dir, 'dev')
        process_dataset(ami_10s_dir_train, ami_400ms_dir_train)
        process_dataset(ami_10s_dir_dev, ami_400ms_dir_dev)
