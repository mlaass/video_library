#!/bin/bash

#==============================================================================
#
#          FILE:  compress.sh
#
#         USAGE:  ./compress.sh input_video [target_percentage]
#
#   DESCRIPTION:  Compresses a video file to reduce its size using one of two methods:
#                 1. Interactive mode: Analyzes the file and offers compression
#                    options (50%, 25%, 10% of original size).
#                 2. Direct mode: Compresses to a specified target percentage.
#
#       OPTIONS:  See function show_help()
#  REQUIREMENTS:  ffmpeg, ffprobe, bc
#          BUGS:  ---
#         NOTES:  Uses CRF (Constant Rate Factor) and resolution scaling for compression
#        AUTHOR:  Your Name
#       VERSION:  1.0
#       CREATED:  $(date +%Y-%m-%d)
#      REVISION:  ---
#
#==============================================================================

# --- Display Help Function ---
show_help() {
    echo "A versatile video compression script using ffmpeg."
    echo ""
    echo "----------------------------------------------------------------------"
    echo "Usage: $0 input_video_file [target_percentage]"
    echo "----------------------------------------------------------------------"
    echo ""
    echo "This script has two modes of operation:"
    echo ""
    echo "  1. Interactive Mode (default)"
    echo "     Analyzes the input file and presents compression options."
    echo ""
    echo "     Arguments:"
    echo "       input_video_file    Path to the video file to compress."
    echo ""
    echo "     Examples:"
    echo "       $0 myvideo.mp4       # Shows interactive compression options"
    echo ""
    echo "  2. Direct Mode"
    echo "     Compresses directly to a specified target percentage."
    echo ""
    echo "     Arguments:"
    echo "       input_video_file    Path to the video file to compress."
    echo "       target_percentage   Target size as percentage of original (10-90)."
    echo ""
    echo "     Examples:"
    echo "       $0 myvideo.mp4 50    # Compress to ~50% of original size"
    echo "       $0 myvideo.mp4 25    # Compress to ~25% of original size"
    echo "       $0 myvideo.mp4 10    # Compress to ~10% of original size"
    echo ""
    echo "Note: Actual compression results may vary based on video content."
    echo ""
    exit 1
}

# --- Function to format file size ---
format_size() {
    local size=$1
    if (( size >= 1073741824 )); then
        echo "$(echo "scale=2; $size / 1073741824" | bc) GB"
    elif (( size >= 1048576 )); then
        echo "$(echo "scale=2; $size / 1048576" | bc) MB"
    elif (( size >= 1024 )); then
        echo "$(echo "scale=2; $size / 1024" | bc) KB"
    else
        echo "$size bytes"
    fi
}

# --- Function to get compression settings ---
get_compression_settings() {
    local target_percent=$1
    local resolution=$2
    
    # Base settings
    local crf=""
    local scale=""
    local preset="medium"
    
    # Determine CRF and scaling based on target percentage
    case $target_percent in
        50)
            crf=28
            # Keep original resolution for 50%
            scale="$resolution"
            ;;
        25)
            crf=32
            # Reduce resolution to ~70% for 25%
            if [[ $resolution =~ ([0-9]+)x([0-9]+) ]]; then
                width=${BASH_REMATCH[1]}
                height=${BASH_REMATCH[2]}
                new_width=$(echo "scale=0; $width * 0.7 / 2" | bc)
                new_width=$((new_width * 2))  # Ensure even number
                scale="${new_width}:-2"
            else
                scale="$resolution"
            fi
            ;;
        10)
            crf=35
            preset="faster"
            # Reduce resolution to ~50% for 10%
            if [[ $resolution =~ ([0-9]+)x([0-9]+) ]]; then
                width=${BASH_REMATCH[1]}
                height=${BASH_REMATCH[2]}
                new_width=$(echo "scale=0; $width * 0.5 / 2" | bc)
                new_width=$((new_width * 2))  # Ensure even number
                scale="${new_width}:-2"
            else
                scale="$resolution"
            fi
            ;;
        *)
            # Custom percentage
            if (( target_percent >= 40 )); then
                crf=28
                scale="$resolution"
            elif (( target_percent >= 20 )); then
                crf=32
                if [[ $resolution =~ ([0-9]+)x([0-9]+) ]]; then
                    width=${BASH_REMATCH[1]}
                    new_width=$(echo "scale=0; $width * 0.8 / 2" | bc)
                    new_width=$((new_width * 2))
                    scale="${new_width}:-2"
                else
                    scale="$resolution"
                fi
            else
                crf=35
                preset="faster"
                if [[ $resolution =~ ([0-9]+)x([0-9]+) ]]; then
                    width=${BASH_REMATCH[1]}
                    new_width=$(echo "scale=0; $width * 0.6 / 2" | bc)
                    new_width=$((new_width * 2))
                    scale="${new_width}:-2"
                else
                    scale="$resolution"
                fi
            fi
            ;;
    esac
    
    echo "$crf|$scale|$preset"
}

# --- Function to compress video ---
compress_video() {
    local input="$1"
    local output="$2"
    local settings="$3"
    
    IFS='|' read -r crf scale preset <<< "$settings"
    
    echo "Compressing with CRF=$crf, Scale=$scale, Preset=$preset..."
    
    if [[ "$scale" == *"x"* ]]; then
        # No scaling needed
        ffmpeg -i "$input" -c:v libx264 -crf "$crf" -preset "$preset" -c:a aac -b:a 128k -y "$output"
    else
        # Apply scaling
        ffmpeg -i "$input" -c:v libx264 -crf "$crf" -preset "$preset" -vf "scale=$scale" -c:a aac -b:a 128k -y "$output"
    fi
}

# --- Main Script Logic ---

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it to use this script."
    exit 1
fi

# Check if ffprobe is installed
if ! command -v ffprobe &> /dev/null; then
    echo "Error: ffprobe is not installed. Please install it to use this script."
    exit 1
fi

# Check if bc is installed
if ! command -v bc &> /dev/null; then
    echo "Error: bc is not installed. Please install it to use this script."
    exit 1
fi

# Check if at least an input file was provided
if [ $# -eq 0 ]; then
    show_help
fi

# Get the input file from the first argument
input_file="$1"
target_percentage="$2"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' not found."
    exit 1
fi

# Get file extension and base name
filename=$(basename -- "$input_file")
extension="${filename##*.}"
basename="${filename%.*}"

# Get file size
file_size=$(stat -f%z "$input_file" 2>/dev/null || stat -c%s "$input_file" 2>/dev/null)
if [ -z "$file_size" ]; then
    echo "Error: Could not determine file size."
    exit 1
fi

# Get video information
echo "Analyzing video file..."
video_info=$(ffprobe -v quiet -print_format json -show_format -show_streams "$input_file")

# Extract resolution
resolution=$(echo "$video_info" | grep -m1 '"width"' | sed 's/[^0-9]*//g')
height=$(echo "$video_info" | grep -m1 '"height"' | sed 's/[^0-9]*//g')
if [ -n "$resolution" ] && [ -n "$height" ]; then
    resolution="${resolution}x${height}"
else
    resolution="unknown"
fi

# Extract duration
duration=$(echo "$video_info" | grep '"duration"' | head -1 | sed 's/[^0-9.]*//g')

# Display file information
echo ""
echo "========================================================================"
echo "File Information:"
echo "========================================================================"
echo "File: $filename"
echo "Size: $(format_size $file_size)"
echo "Resolution: $resolution"
if [ -n "$duration" ]; then
    echo "Duration: $(echo "scale=1; $duration / 60" | bc) minutes"
fi
echo ""

# --- Mode Detection ---
if [ -n "$target_percentage" ]; then
    # --- DIRECT MODE ---
    
    # Validate percentage
    if ! [[ "$target_percentage" =~ ^[0-9]+$ ]] || [ "$target_percentage" -lt 10 ] || [ "$target_percentage" -gt 90 ]; then
        echo "Error: Target percentage must be a number between 10 and 90."
        exit 1
    fi
    
    output_file="${basename}_${target_percentage}percent.${extension}"
    
    echo "Direct compression mode: targeting ~${target_percentage}% of original size"
    echo "Output file: $output_file"
    echo ""
    
    # Get compression settings
    settings=$(get_compression_settings "$target_percentage" "$resolution")
    
    # Compress the video
    compress_video "$input_file" "$output_file" "$settings"
    
else
    # --- INTERACTIVE MODE ---
    
    echo "Select compression target:"
    echo "1) 50% of original size (~$(format_size $(echo "$file_size * 0.5 / 1" | bc)))"
    echo "2) 25% of original size (~$(format_size $(echo "$file_size * 0.25 / 1" | bc)))"
    echo "3) 10% of original size (~$(format_size $(echo "$file_size * 0.10 / 1" | bc)))"
    echo "4) Custom percentage"
    echo "5) Cancel"
    echo ""
    
    while true; do
        read -p "Enter your choice (1-5): " choice
        case $choice in
            1)
                target_percentage=50
                break
                ;;
            2)
                target_percentage=25
                break
                ;;
            3)
                target_percentage=10
                break
                ;;
            4)
                while true; do
                    read -p "Enter target percentage (10-90): " custom_percent
                    if [[ "$custom_percent" =~ ^[0-9]+$ ]] && [ "$custom_percent" -ge 10 ] && [ "$custom_percent" -le 90 ]; then
                        target_percentage=$custom_percent
                        break 2
                    else
                        echo "Please enter a valid percentage between 10 and 90."
                    fi
                done
                ;;
            5)
                echo "Operation cancelled."
                exit 0
                ;;
            *)
                echo "Invalid choice. Please enter 1-5."
                ;;
        esac
    done
    
    output_file="${basename}_${target_percentage}percent.${extension}"
    
    echo ""
    echo "Compressing to ~${target_percentage}% of original size..."
    echo "Output file: $output_file"
    echo ""
    
    # Get compression settings
    settings=$(get_compression_settings "$target_percentage" "$resolution")
    
    # Compress the video
    compress_video "$input_file" "$output_file" "$settings"
fi

# Check if ffmpeg command was successful
if [ $? -eq 0 ]; then
    # Get output file size
    output_size=$(stat -f%z "$output_file" 2>/dev/null || stat -c%s "$output_file" 2>/dev/null)
    if [ -n "$output_size" ]; then
        actual_percentage=$(echo "scale=1; $output_size * 100 / $file_size" | bc)
        echo ""
        echo "========================================================================"
        echo "Compression Complete!"
        echo "========================================================================"
        echo "Original size: $(format_size $file_size)"
        echo "Compressed size: $(format_size $output_size)"
        echo "Actual compression: ${actual_percentage}% of original"
        echo "Space saved: $(format_size $(echo "$file_size - $output_size" | bc))"
        echo "Output saved as: '$output_file'"
        echo ""
    else
        echo "Done! Output saved as '$output_file'"
    fi
else
    echo "An error occurred during the compression process."
    # Clean up partially created file on error
    rm -f "$output_file"
    exit 1
fi