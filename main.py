import os
from Haze import HazeRemover
from dark_channer_prior import Dark_channel_prior



if __name__ == '__main__':
    dehazer = Dark_channel_prior(input_folder='input_folder', output_folder='output_folder')
    dehazer.process_images()


# remover = HazeRemover("input_folder", 'output_folder')
# remover.process_images()

# remove = Dark_channel_prior("input_folder", "output_folder")
# for dark_channer in remove.process_images():
#     dark_channer