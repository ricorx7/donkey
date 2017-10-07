""""
I want to start using 1/r for my steering angles
When using in the training
This will rescale all the old data stored in data
so that the steering angle is 1/r.

It is suggest to invert the steering angle when running through the CNN.
We use 1/r instead of r to prevent a singularity when driving straight
(the turning radius for driving straight is infinity).
1/r smoothly transitions through zero from left turns (negative values)
to right turns (positive values).

Usage:
  rescale_steering_angle PATH [--output=<output>]

"""
import json as json
import glob
import os
from shutil import copy2
from docopt import docopt


def main(args):

    # Folder path to modify all the data
    #folder_path = '/Users/rico/ricar_donkey/rpi/data/tub_65_17-08-27'
    folder_path = args['PATH']

    # Output folder
    if args['--output']:
        output_folder_path = args['--output']
    else:
        output_folder_path = os.path.join(folder_path, 'inverted')

    if not os.path.exists(folder_path):
        print("Folder does not exist")
        exit(-1)

    # Create a new folder so old data is maintained
    if os.path.exists(output_folder_path):
        print("Output folder already exists. Use --output to change the folder path.")
        exit(-1)
    else:
        os.mkdir(output_folder_path)

    print("Output Folder: ", output_folder_path)

    # Get a lit of all the records
    json_file_list = glob.glob(os.path.join(folder_path,  "record_*.json"))

    if len(json_file_list) > 0:

        for file in json_file_list:
            try:
                # Read in the json files except meta.json
                with open(file, 'r') as data_file:
                    data = json.load(data_file)

                    # Invert the angle
                    if(data["user/angle"]) != 0:
                        data["user/angle"] = 1 / data["user/angle"]

                    # Write the new JSON file
                    file_name = os.path.basename(file)
                    new_file_path = os.path.join(output_folder_path, file_name)
                    with open(new_file_path, 'w') as f:
                        json.dump(data, f)

                    # Copy the image
                    copy2(os.path.join(folder_path, data["cam/image_array"]), output_folder_path)
            except json.decoder.JSONDecodeError:
                print("File bad: " + file)

        # Copy meta.json
        copy2(os.path.join(folder_path, 'meta.json'), output_folder_path)


if __name__ == "__main__":
    args = docopt(__doc__, version='Rescale Version 1.0.0')

    main(args)
