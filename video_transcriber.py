import os
import argparse
from pathlib import Path
from faster_whisper import WhisperModel
import ffmpeg


def get_whisper_model(model_size="medium", device=None):
    """
    Initializes and returns the Whisper model.
    This function checks for the best available device (CUDA or CPU)
    and loads the specified Faster Whisper model.
    """
    if device is None:
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                print("CUDA is available. Using GPU for transcription.")
            else:
                device = "cpu"
                print("CUDA not available. Using CPU for transcription.")
        except ImportError:
            device = "cpu"
            print("PyTorch not found. Using CPU for transcription.")

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading Whisper model (size: {model_size}, device: {device}, compute_type: {compute_type})")

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have the correct dependencies installed.")
        return None

    return model


def format_timestamp(seconds):
    """
    Converts seconds into a H:M:S,ms format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}]"


def transcribe_video(video_path, model, output_dir):
    """
    Transcribes a single video file and saves the transcript with timestamps.
    """
    if not video_path.is_file():
        print(f"Error: Video file not found at {video_path}")
        return

    print(f"\nProcessing: {video_path.name}")

    # Define the output path for the transcript
    transcript_filename = f"{video_path.stem}_transcript.md"
    transcript_path = output_dir / transcript_filename

    if transcript_path.exists():
        print(f"Transcript already exists for {video_path.name}. Skipping.")
        return

    try:
        # Transcribe with word-level timestamps
        segments, _ = model.transcribe(str(video_path), word_timestamps=True)

        # Process segments to create transcript with timestamps every 10 seconds
        with open(transcript_path, "w", encoding="utf-8") as f:
            current_time = 0
            line_buffer = ""

            print("Transcription in progress...")
            for segment in segments:
                for word in segment.words:
                    if not line_buffer:
                        # Start a new line with a timestamp
                        f.write(f"{format_timestamp(word.start)} ")
                        current_time = word.start

                    line_buffer += word.word + " "

                    # If the current line exceeds 10 seconds, start a new one
                    if word.end - current_time > 10.0:
                        f.write(line_buffer.strip() + "\n")
                        line_buffer = ""

            # Write any remaining text in the buffer
            if line_buffer:
                f.write(line_buffer.strip())

            print(f"✅ Successfully transcribed {video_path.name}")
            print(f"   Transcript saved to: {transcript_path}")

    except ffmpeg.Error as e:
        print(f"ffmpeg error processing {video_path.name}: {e.stderr.decode()}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {video_path.name}: {e}")


def main():
    """
    Main function to parse arguments and process videos.
    """
    parser = argparse.ArgumentParser(description="Transcribe all video files in a folder using Faster Whisper.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing video files.")
    parser.add_argument(
        "--output_folder", type=str, help="Path to the folder to save transcripts. Defaults to the input folder."
    )
    parser.add_argument(
        "--model_size", type=str, default="medium", help="Whisper model size (e.g., tiny, base, small, medium, large)."
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use for the Whisper model (e.g., cuda, cpu). Auto-detects if not specified.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_folder)
    output_path = Path(args.output_folder) if args.output_folder else input_path

    # Validate input directory
    if not input_path.is_dir():
        print(f"Error: Input folder not found at {input_path}")
        return

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the Whisper model
    model = get_whisper_model(args.model_size, args.device)
    if model is None:
        return

    # Supported video file extensions
    video_extensions = [".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm"]

    # Iterate through all files in the input folder
    print(f"\nStarting transcription process for videos in: {input_path}")
    for file_path in input_path.iterdir():
        if file_path.suffix.lower() in video_extensions:
            transcribe_video(file_path, model, output_path)
        else:
            print(f"Skipping non-video file: {file_path.name}")

    print("\n✅ All videos have been processed.")


if __name__ == "__main__":
    main()
