"""
Assemble pictures in a folder, write to a videofile and gif
Convert the folder of images to a video.
Usage:
  convert_images_to_video.py PATH [--output=<output_folder>] [--video_name=<video_name>]

"""
from moviepy.editor import ImageSequenceClip
import os
from shutil import copy2, rmtree
from tqdm import tqdm           # Show a progress bar
import glob
import cv2
from donkeycar.lane_detect.image_pipeline import pipeline

def main(args):

    # Draw lane lines in the pictures
    draw_lane_lines = True

    img_folder = args["PATH"]
    if not os.path.exists(args["PATH"]):
        print("Image folder does not exist: ", img_folder)
        exit(-1)

    new_img_folder = os.path.join(img_folder, "video")
    if not os.path.exists(new_img_folder):
        os.mkdir(os.path.join(img_folder, "video"))
    else:
        print("New Image Folder already exists.", new_img_folder)
        exit(-2)

    output_folder = new_img_folder
    if args["--output"]:
        if not os.path.exists(args["--output"]):
            output_folder = args["--output"]
        else:
            print("Given Output Folder does not exist: ", args["--output"])

    video_name = "output.mp4"
    if args["--video_name"]:
        video_name = args["--video_name"]

    output_path = os.path.join(output_folder, video_name)

    print("Image Folder: ", img_folder)
    print("New Image Folder: ", new_img_folder)
    print("Output: ", output_path)

    # Copy all the images from the image folder and paste it into a new folder
    # So that it is only images
    img_file_list = glob.glob(os.path.join(img_folder, "*.jpg"))
    for img in tqdm(img_file_list):
        if draw_lane_lines:
            lane_line_img = pipeline(img)                         # Set the line detection pipeline
            #print(os.path.join(new_img_folder, os.path.basename(img)))
            cv2.imwrite(os.path.join(new_img_folder, os.path.basename(img)), lane_line_img)
        else:
            copy2(img, new_img_folder)

    clip = ImageSequenceClip(new_img_folder, fps=20)
    clip.write_videofile(output_path, fps=20, codec='libx264') # many options available

    # Copy the video to the original folder
    # Remove the folder created
    copy2(output_path, img_folder)
    #rmtree(new_img_folder)


if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__, version='Images to Video 1.0.0')
    main(args)
