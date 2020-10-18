import os


def clean_txt(txt_path):
    f = open(txt_path, 'r')
    lines = f.readlines()
    cleaned = [str(0) + line[1:] for line in lines]

    with open(txt_path, 'w') as f:
        for line in cleaned:
            f.write(line)


def main():
    directory = 'yolov5/data_train_yolo'
    for file in os.listdir(directory):
        if file[-4:] == '.txt' and file != 'classes.txt':
            clean_txt('/'.join([directory, file]))
        else:
            continue


if __name__ == "__main__":
    main()
