from shutil import copyfile
from  pathlib import Path
from random import randrange
import argparse


def create_subset(lighting_condition:str, data_dir:str):

    DATA_DIR = Path(data_dir)

    OUTPUT_DIR = Path(f'{DATA_DIR.name}_selection_{lighting_condition}/images')
    if not OUTPUT_DIR.is_dir():
        OUTPUT_DIR.mkdir(parents=True)

    if lighting_condition == 'random':
        num_images = len(list(DATA_DIR.glob('*_0_*')))
        files = []
        for i in range(1, num_images+1):
            files.extend(sorted(DATA_DIR.glob(f'*_{str(i).zfill(3)}_{randrange(7)}*')))
    
    else:
        files = list(DATA_DIR.glob(f'*_{lighting_condition}_*'))

    for file in files:
        copyfile(file, OUTPUT_DIR / file.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create subsets of data.')
    parser.add_argument('-l', '--lighting', type=str,
                    help='choose a lighting condition',
                    choices=['1','2','3','4','5','6','max','random'],
                    required=True)
    parser.add_argument('--data', type=str, default='/mnt/raid/jpencharz/mvs_clean_scan1',
                    help='specifiy data directory',
                    required=False)
    
    args = parser.parse_args()

    create_subset(lighting_condition=args.lighting, data_dir=args.data)




