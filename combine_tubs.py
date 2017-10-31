""""
Combine all the tubs i want to use into 1 folder.
I have found that when i run multiple folders, my loss
plot spikes randomly.  I have been told this happens
when i jump between folders.  So this will make all the tubs
1 large tub.

It will rename the json and image files so that everything
is in order.

Usage:
  combine_tubs.py --input=<tubs> --output=<output> [--force]

"""
import json as json
import glob
import os
from shutil import copy2, rmtree
from docopt import docopt


def main(args):
    # Folders to combine
    # Separated commas
    folder_paths = args['--input']

    # Output folder
    output_folder_path = args['--output']

    # Check if the folder already exists
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    else:
        if args['--force']:
            #os.rmdir(output_folder_path)
            rmtree(output_folder_path)
            os.mkdir(output_folder_path)
        else:
            print("Folder already exist, please give a new folder")
            exit(-1)

    print(os.listdir(folder_paths))
    print('Number of Tubs: ', get_num_tubs(folder_paths))

    index = 1
    for tub in get_tubs(folder_paths):
        print(get_num_records(tub))
        index = copy_tub(tub, index, output_folder_path)


def copy_tub(path, index, dest):
    tub_records = get_records(path)

    # Go there each record, copying the files and modifying the json image path
    for record in tub_records:
        copy_tub_record(record, index, dest)
        index += 1

    return index


def copy_tub_record(file, index, dest):
    # Generate new file name based off index
    new_record_name = 'record_' + str(index).zfill(8) + '.json'
    new_image_name = str(index).zfill(8) + '_cam-image_array_.jpg'

    old_folder_path = os.path.dirname(file)

    # copy the file to the destination
    copy2(file, os.path.join(dest, new_record_name))

    # copy the image file to the destination
    orig_img = get_record_image(file)
    if not orig_img:
        return

    copy2(os.path.join(old_folder_path, orig_img), os.path.join(dest, new_image_name))

    # modify the json with the new image file name
    try:
        # Read in the json file
        with open(os.path.join(dest, new_record_name), 'r') as data_file:
            data = json.load(data_file)

        # Write the new JSON file
        with open(os.path.join(dest, new_record_name), 'w') as f:
            data["cam/image_array"] = new_image_name
            json.dump(data, f)

    except json.decoder.JSONDecodeError:
        print("File bad: " + file)


def get_record_image(record_path):
    try:
        with open(record_path, 'r') as data_file:
            data = json.load(data_file)

            return data["cam/image_array"]
    except json.decoder.JSONDecodeError:
        print("File bad: " + record_path)

def get_tubs(path):
    return glob.glob(os.path.join(path, 'tub_*'))


def get_num_tubs(path):
    return len(get_tubs(path))


def get_records(path):
    return glob.glob(os.path.join(path, 'record_*.json'))


def get_num_records(path):
    return len(get_records(path))





if __name__ == "__main__":
    args = docopt(__doc__, version='Combine Tubs 1.0.0')

    main(args)

