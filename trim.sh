#!/bin/bash

#==============================================================================
#
#          FILE:  trim.sh
#
#         USAGE:  ./trim.sh input_video [start] [end]
#
#   DESCRIPTION:  Trims a video file using one of two methods:
#                 1. By duration: Trims a specified number of seconds from the
#                    start and end of the video.
#                 2. By timestamp: Extracts a clip between a specific start
#                    and end time.
#
#       OPTIONS:  See function show_help()
#  REQUIREMENTS:  ffmpeg, ffprobe, bc
#          BUGS:  ---
#         NOTES:  ---
#        AUTHOR:  Gemini
#       VERSION:  2.0
#       CREATED:  2024-05-21
#      REVISION:  ---
#
#==============================================================================

# --- Display Help Function ---
show_help() {
    echo "A versatile video trimming script using ffmpeg."
    echo ""
    echo "----------------------------------------------------------------------"
    echo "Usage: $0 input_video_file [start_time] [end_time]"
    echo "----------------------------------------------------------------------"
    echo ""
    echo "This script has two modes of operation:"
    echo ""
    echo "  1. Trim by Duration (cutting from ends)"
    echo "     If start/end times are provided as plain numbers (seconds)."
    echo ""
    echo "     Arguments:"
    echo "       input_video_file    Path to the video file to trim."
    echo "       start_trim          Seconds to trim from the start (default: 3)."
    echo "       end_trim            Seconds to trim from the end (default: 3)."
    echo ""
    echo "     Examples:"
    echo "       $0 myvideo.mp4           # Trim 3s from start and 3s from end."
    echo "       $0 myvideo.mp4 5         # Trim 5s from start and 3s from end."
    echo "       $0 myvideo.mp4 10 5      # Trim 10s from start and 5s from end."
    echo ""
    echo "  2. Trim by Timestamp (extracting a clip)"
    echo "     If start/end times are provided in [HH:]MM:SS format."
    echo ""
    echo "     Arguments:"
    echo "       input_video_file    Path to the video file to trim."
    echo "       start_time          The start timestamp of the desired clip."
    echo "       end_time            The end timestamp of the desired clip."
    echo ""
    echo "     Examples:"
    echo "       $0 myvideo.mp4 01:30 02:45      # Clip from 1m 30s to 2m 45s."
    echo "       $0 myvideo.mp4 00:05:10 00:08:00 # Clip from 5m 10s to 8m 0s."
    echo ""
    exit 1
}

# --- Main Script Logic ---

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it to use this script."
    exit 1
fi

# Check if at least an input file was provided
if [ $# -eq 0 ]; then
    show_help
fi

# Get the input file from the first argument
input_file="$1"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' not found."
    exit 1
fi

# Get file extension and base name
filename=$(basename -- "$input_file")
extension="${filename##*.}"
basename="${filename%.*}"

# Create output filename
output_file="${basename}_clipped.${extension}"

# --- Mode Detection ---
# Check if the second argument contains a colon to decide the mode.
# If no second argument exists, it will default to the 'else' block (duration mode).
if [[ "$2" == *":"* ]]; then
    # --- MODE 2: Trim by Timestamp ---
    start_time="$2"
    end_time="$3"

    # Check that an end time was provided for this mode
    if [ -z "$end_time" ]; then
        echo "Error: An end timestamp is required for this mode."
        show_help
    fi

    echo "Processing '$input_file' (Timestamp Mode)..."
    echo "Extracting clip from $start_time to $end_time..."

    # Run ffmpeg command. It directly accepts HH:MM:SS format.
    # This command re-encodes the video to ensure the cut is precise.
    ffmpeg -i "$input_file" -ss "$start_time" -to "$end_time" -c:v libx264 -c:a aac -strict experimental -y "$output_file"

else
    # --- MODE 1: Trim by Duration ---
    # Set default values for trim amounts if not provided
    start_trim=${2:-3}
    end_trim=${3:-3}

    echo "Processing '$input_file' (Duration Mode)..."
    echo "Trimming $start_trim seconds from start and $end_trim seconds from end..."

    # Get video duration using ffprobe
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input_file")
    if [ -z "$duration" ]; then
        echo "Error: Could not determine video duration."
        exit 1
    fi

    # Calculate the 'to' time for ffmpeg by subtracting the end trim from the total duration
    end_time_calc=$(echo "$duration - $end_trim" | bc)

    # Run ffmpeg command with re-encoding to ensure an accurate start point
    # -ss is placed before -i for slow but precise seeking.
    ffmpeg -i "$input_file" -ss "$start_trim" -to "$end_time_calc" -c:v libx264 -c:a aac -strict experimental -y "$output_file"

    # Alternative faster method (stream copy):
    # This is much faster but may be less accurate as it cuts on keyframes.
    # Uncomment the line below and comment the one above to use it.
    # ffmpeg -ss "$start_trim" -i "$input_file" -to $(echo "$duration - $start_trim - $end_trim" | bc) -c copy -y "$output_file"

fi

# Check if ffmpeg command was successful
if [ $? -eq 0 ]; then
    echo "Done! Output saved as '$output_file'"
else
    echo "An error occurred during the ffmpeg process."
    # Clean up partially created file on error
    rm -f "$output_file"
    exit 1
fi
