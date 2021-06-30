def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed

# data/augmentation/images_txt_source/ffd6378b78d84c2df418d855a3b3f7565bdb07044bac91233b6d59ba0837039c.jpg
