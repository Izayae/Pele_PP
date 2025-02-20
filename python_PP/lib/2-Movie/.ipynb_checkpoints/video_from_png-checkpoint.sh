#!/bin/bash

# load ffmpeg
module load ffmpeg

# parameters
custom_name="multiplot"
image_dir="/project/b/bsavard/mavab/public_Post-processing/PP_python/Results/images/tuto_bunsen/contours/"
video_dir="/project/b/bsavard/mavab/public_Post-processing/PP_python/Results/video/tuto_bunsen/contours/"
mkdir -p ${video_dir}
video_file="${custom_name}.mp4"

# Create videos
ffmpeg -framerate 4 -pattern_type glob -i "${image_dir}/${custom_name}_*.png" -c:v libx264 -pix_fmt yuv420p ${video_dir}/${video_file}