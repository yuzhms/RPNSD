import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--split')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    split = args.split
    output_path = os.path.abspath(os.path.join(args.output_dir, split))
    os.makedirs(output_path, exist_ok=True)
    input_path = os.path.abspath(os.path.join(args.data_dir, 'amicorpus'))
    scp_file = os.path.join(output_path, 'wav.scp')
    utt2spk_file = os.path.join(output_path, 'utt2spk')
    list_file = os.path.join(args.data_dir, 'AMI', f'MixHeadset.{split}.lst')
    with open(list_file) as f:
        file_list = [_.strip() for _ in f.readlines()]
    scp_lines = []
    utt2spk_lines = []
    for file in file_list:
        abs_file_name = os.path.join(input_path, file.split('.')[0], 'audio', file+'.wav')
        scp_lines.append(f'{file} {abs_file_name}\n')
        utt2spk_lines.append(f'{file} {file}\n')
    with open(scp_file, 'w') as f:
        f.writelines(scp_lines)
    with open(utt2spk_file, 'w') as f:
        f.writelines(utt2spk_lines)



