from pathlib import Path
from numpy import loadtxt


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def create_txt_labels(images_dir, images_ext, txt_source_dir, txt_result_dir, image_size):
    images_list = get_all_files_in_folder(images_dir, images_ext)
    txt_source_list = get_all_files_in_folder(txt_source_dir, ('*.txt'))

    # create txt_result folder if not exist
    Path(txt_result_dir).mkdir(parents=True, exist_ok=True)

    for image in images_list:
        filename = image.stem

        for txt in txt_source_list:
            if txt.stem == filename:
                lines = loadtxt(str(Path(txt_source_dir).joinpath(txt.name)), delimiter=" ", unpack=False)

                with open(Path(txt_result_dir).joinpath(txt.name), 'w') as f:
                    for item in lines:
                        width = int(item[3] * image_size)
                        height = int(item[4] * image_size)

                        x = int(item[1] * image_size - width / 2)
                        y = int(item[2] * image_size - height / 2)

                        label = 0
                        rec = str(label) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height)
                        f.write("%s\n" % rec)


if __name__ == '__main__':
    image_size = 1024
    images_dir = Path('data/evaluation/images')
    images_ext = ['*.jpg']
    txt_source_dir = Path('data/evaluation/txt_source')
    txt_result_dir = Path('data/evaluation/txt_result')

    create_txt_labels(images_dir=images_dir,
                      images_ext=images_ext,
                      txt_source_dir=txt_source_dir,
                      txt_result_dir=txt_result_dir,
                      image_size=image_size)
