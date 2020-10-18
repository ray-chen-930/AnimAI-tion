import os
import shutil
import random


def get_train_val_names(li, train_perc=.8):
    l_len = len(li)
    idx_cut = int(train_perc*l_len)
    random.shuffle(li)
    train_l = li[:idx_cut]
    val_l = li[idx_cut:]

    return train_l, val_l


def main():
    directory = 'yolov5/data_train_yolo'
    im_train = '/'.join([directory, 'images_for_train', 'train'])
    im_val = '/'.join([directory, 'images_for_train', 'val'])
    lab_train = '/'.join([directory, 'labels', 'train'])
    lab_val = '/'.join([directory, 'labels', 'val'])

    all_names = []
    for file in os.listdir(directory):
        if file[-4:] == '.txt' and file != 'classes.txt':
            all_names.append(file[:-4])

    train_l, val_l = get_train_val_names(all_names)

    for i in train_l:
        file_path = '/'.join([directory, i])
        shutil.move(file_path + '.txt', lab_train)
        if i + '.jpg' in os.listdir(directory):
            shutil.move(file_path + '.jpg', im_train)

        if i + '.jpeg' in os.listdir(directory):
            shutil.move(file_path + '.jpeg', im_train)

        if i + '.png' in os.listdir(directory):
            shutil.move(file_path + '.png', im_train)

    for i in val_l:
        file_path = '/'.join([directory, i])
        shutil.move(file_path + '.txt', lab_val)
        if i + '.jpg' in os.listdir(directory):
            shutil.move(file_path + '.jpg', im_val)

        if i + '.jpeg' in os.listdir(directory):
            shutil.move(file_path + '.jpeg', im_val)

        if i + '.png' in os.listdir(directory):
            shutil.move(file_path + '.png', im_val)


if __name__ == "__main__":
    main()
