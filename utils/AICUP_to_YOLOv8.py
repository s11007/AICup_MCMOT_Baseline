import os
import glob
import shutil
import argparse
from tqdm import tqdm

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--AICUP_dir', type=str, default='', help='your AICUP dataset path')
    parser.add_argument('--YOLOv8_dir', type=str, default='', help='converted dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='The ratio of the train set when splitting the train set and the validation set')
    opt = parser.parse_args()
    return opt


def aicup_to_yolo(args):
    train_dir = os.path.join(args.YOLOv8_dir, 'train')
    valid_dir = os.path.join(args.YOLOv8_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    
    if os.path.exists(valid_dir):
        shutil.rmtree(valid_dir)

    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    
    os.makedirs(os.path.join(valid_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, 'labels'), exist_ok=True)
    
    # Collect image files
    image_files = []
    for root, _, files in os.walk(os.path.join(args.AICUP_dir, 'images')):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    # Collect label files
    label_files = []
    for root, _, files in os.walk(os.path.join(args.AICUP_dir, 'labels')):
        for file in files:
            if file.endswith('.txt'):
                label_files.append(os.path.join(root, file))
    
    total_count = len(image_files)
    train_count = int(total_count * args.train_ratio)

    train_files = image_files[:train_count]
    valid_files = image_files[train_count:]
    
    for src_path in tqdm(train_files + valid_files, desc=f'copying data'):
        text = src_path.split('\\')
        timestamp = text[-2]
        camID_frameID = text[-1]
        
        train_or_valid = 'train' if src_path in train_files else 'valid'
        
        dst_image_path = os.path.join(args.YOLOv8_dir, train_or_valid, 'images', timestamp + '_' + camID_frameID)
        dst_label_path = os.path.join(args.YOLOv8_dir, train_or_valid, 'labels', timestamp + '_' + camID_frameID.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        
        shutil.copy2(src_path, dst_image_path)
        
        # Find corresponding label file
        label_file = src_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        if label_file in label_files:
            shutil.copy2(label_file, dst_label_path)
    
    return 0

def delete_track_id(labels_dir):
    for file_path in tqdm(glob.glob(os.path.join(labels_dir, '*.txt')), desc='delete_track_id'):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            text = line.split(' ')
            
            if len(text) > 5:
                new_lines.append(line.replace(' ' + text[-1], '\n'))

        with open(file_path, 'w') as f:
            f.writelines(new_lines)

    return 0

if __name__ == '__main__':
    args = arg_parse()
    
    # debug
    # args.AICUP_dir = '/mnt/Nami/AI_CUP_MCMOT_dataset/train'
    # args.YOLOv8_dir = '/mnt/Nami/AI_CUP_MCMOT_dataset/yolo'
    # args.train_ratio = 0.8
    
    aicup_to_yolo(args)
    
    train_dir = os.path.join(args.YOLOv8_dir, 'train', 'labels')
    val_dir = os.path.join(args.YOLOv8_dir, 'valid', 'labels')
    delete_track_id(train_dir)
    delete_track_id(val_dir)
