#!/usr/bin/env python3
"""
Video Viewer Flask Application
A simple web server to view videos with transcript integration
"""

import os
import re
import subprocess
import threading
import time
from pathlib import Path
from flask import Flask, render_template, send_file, url_for, request, jsonify
import json
import ffmpeg
from faster_whisper import WhisperModel
import requests
from dotenv import load_dotenv

FACE_RECOGNITION_ENABLED = True

try:
    from face_recognizer import (
        FaceLibrary,
        FaceRecognizer,
        process_video_face_recognition,
        load_face_recognition_data,
        save_face_recognition_data,
    )
except Exception as e:
    print(f"Face recognition disabled (import error): {e}")
    FACE_RECOGNITION_ENABLED = False

    # Provide safe fallbacks so the rest of the app can run
    FaceLibrary = None

    def process_video_face_recognition(video_path, task_id, processing_status):
        processing_status[task_id] = {
            "status": "error",
            "message": "Face recognition is disabled (missing dependencies)",
        }

    def load_face_recognition_data(video_path):
        return None

    def save_face_recognition_data(video_path, face_data):
        return None


app = Flask(__name__)

# Configuration
VIDEO_DIR = "~/workspace/video_work/screen_recording"  # Update this path
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
METADATA_CACHE_FILE = ".video_metadata_cache.json"

# Global variables for processing status
processing_status = {}
whisper_model = None


def get_video_directory():
    """Get the absolute path to the video directory"""
    return os.path.expanduser(VIDEO_DIR)


def parse_transcript(transcript_path):
    """Parse transcript file and extract title, summary, and timestamped content"""
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        title = None
        summary = None
        transcript_lines = []

        # Look for title (usually first non-empty line or line starting with #)
        for line in lines:
            line = line.strip()
            if line and not line.startswith("[") and not title:
                if line.startswith("#"):
                    title = line.strip("#").strip()
                elif len(line) > 10:  # Reasonable title length
                    title = line
                break

        # Look for summary (usually after title, before timestamped content)
        in_summary = False
        summary_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith("## Summary") or line.startswith("Summary"):
                in_summary = True
                continue
            elif line.startswith("---") or line.startswith("["):
                in_summary = False
            elif in_summary and line:
                summary_lines.append(line)

        summary = " ".join(summary_lines) if summary_lines else None

        # Parse timestamped lines
        timestamp_pattern = r"\[(\d{2}:\d{2}:\d{2}[,\.]\d{3})\]"

        for line in lines:
            line = line.strip()
            match = re.match(timestamp_pattern, line)
            if match:
                timestamp = match.group(1)
                text = line[match.end() :].strip()
                # Convert timestamp to seconds for video seeking
                time_parts = timestamp.replace(",", ".").split(":")
                seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + float(time_parts[2])
                transcript_lines.append({"timestamp": timestamp, "seconds": seconds, "text": text})

        return {"title": title, "summary": summary, "transcript": transcript_lines}

    except Exception as e:
        print(f"Error parsing transcript {transcript_path}: {e}")
        return None


def get_video_info(video_path, metadata_cache):
    """Get video information including transcript and face recognition data if available"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    transcript_path = os.path.join(os.path.dirname(video_path), f"{video_name}_transcript.md")
    faces_path = os.path.join(os.path.dirname(video_path), f"{video_name}_faces.json")

    # Get duration and video date from cache
    duration, video_date = get_cached_metadata(video_path, metadata_cache)

    info = {
        "filename": os.path.basename(video_path),
        "name": video_name,
        "path": video_path,
        "size": os.path.getsize(video_path),
        "modified_time": os.path.getmtime(video_path),
        "video_date": video_date,
        "duration": duration,
        "has_transcript": os.path.exists(transcript_path),
        "has_faces": os.path.exists(faces_path),
    }

    # Load transcript data
    if info["has_transcript"]:
        transcript_data = parse_transcript(transcript_path)
        if transcript_data:
            info.update(transcript_data)
            # Use transcript title if available, otherwise use filename
            if not info.get("title"):
                info["title"] = video_name
        else:
            info["title"] = video_name
    else:
        info["title"] = video_name

    # Load face recognition data (if feature is enabled)
    if FACE_RECOGNITION_ENABLED and info["has_faces"]:
        face_data = load_face_recognition_data(video_path)
        if face_data:
            info["face_data"] = {
                "people_found": face_data.get("people_found", []),
                "total_faces": face_data.get("total_faces", 0),
                "faces_by_timestamp": face_data.get("faces_by_timestamp", {}),
            }

    return info


def get_all_videos():
    """Get all video files from the directory"""
    video_dir = get_video_directory()
    videos = []

    if not os.path.exists(video_dir):
        return videos

    # Load metadata cache
    metadata_cache = load_metadata_cache()
    original_cache_size = len(metadata_cache)

    for filename in os.listdir(video_dir):
        if any(filename.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
            video_path = os.path.join(video_dir, filename)
            video_info = get_video_info(video_path, metadata_cache)
            videos.append(video_info)

    # Save cache if it was updated (new entries added or cache was empty)
    if len(metadata_cache) > original_cache_size or original_cache_size == 0:
        save_metadata_cache(metadata_cache)

    # Sort by video date (newest first) - use extracted date from filename or creation time
    videos.sort(key=lambda x: x["video_date"], reverse=True)
    return videos


def load_metadata_cache():
    """Load video metadata cache from JSON file"""
    video_dir = get_video_directory()
    cache_path = os.path.join(video_dir, METADATA_CACHE_FILE)

    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_metadata_cache(cache):
    """Save video metadata cache to JSON file"""
    video_dir = get_video_directory()
    cache_path = os.path.join(video_dir, METADATA_CACHE_FILE)

    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata cache: {e}")


def get_video_duration(video_path):
    """Get video duration in seconds using ffmpeg"""
    try:
        probe = ffmpeg.probe(video_path)
        return float(probe["streams"][0]["duration"])
    except Exception as e:
        print(f"Error getting duration for {video_path}: {e}")
        return None


def get_cached_metadata(video_path, cache):
    """Get video metadata from cache or extract and cache it"""
    filename = os.path.basename(video_path)
    file_mtime = os.path.getmtime(video_path)

    # Check if we have cached data and if the file hasn't been modified
    if filename in cache:
        cached_data = cache[filename]
        if cached_data.get("mtime") == file_mtime:
            return cached_data.get("duration"), cached_data.get("video_date")

    # Extract duration and video date
    duration = get_video_duration(video_path)

    # Try to extract date from filename, fall back to file creation time
    video_date = extract_date_from_filename(filename)
    if video_date is None:
        # Try to get file birth time (creation time) if available
        try:
            stat = os.stat(video_path)
            # On Linux, try st_birthtime (not always available), fall back to st_ctime
            video_date = getattr(stat, "st_birthtime", stat.st_ctime)
        except (AttributeError, OSError):
            # Fall back to modification time
            video_date = file_mtime

    # Update cache
    cache[filename] = {"duration": duration, "video_date": video_date, "mtime": file_mtime}

    return duration, video_date


def format_duration(seconds):
    """Format duration in seconds to H:MM:SS or M:SS format"""
    if seconds is None:
        return "Unknown"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def extract_date_from_filename(filename):
    """Extract date from filename pattern YYYY-MM-DD_HH-MM-SS or fall back to file creation time"""
    import datetime

    # Try to extract date from filename pattern like 2024-08-29_12-48-34.mp4
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", filename)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        try:
            dt = datetime.datetime(year, month, day, hour, minute, second)
            return dt.timestamp()
        except ValueError:
            # Invalid date in filename, fall back to file stats
            pass

    # Fall back to file creation/birth time if available, otherwise modification time
    return None


# Video Processing Functions
def get_whisper_model_instance(model_size="medium", device=None):
    """Get or create Whisper model instance"""
    global whisper_model

    if whisper_model is None:
        if device is None:
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

        compute_type = "float16" if device == "cuda" else "int8"
        try:
            whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return None

    return whisper_model


def format_timestamp(seconds):
    """Convert seconds to timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}]"


def transcribe_video_file(video_path, output_path, task_id):
    """Transcribe a video file in background with detailed progress tracking"""
    try:
        processing_status[task_id] = {
            "status": "loading_model",
            "progress": 5,
            "details": "Loading Whisper model (base)...",
        }

        # Load Whisper model if not already loaded
        global whisper_model
        if whisper_model is None:
            whisper_model = WhisperModel("base", device="auto", compute_type="auto")

        # Get video duration for progress calculation
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
            duration = float(video_stream.get("duration", 0)) if video_stream else 0
        except:
            duration = 0

        processing_status[task_id] = {
            "status": "transcribing",
            "progress": 15,
            "details": f"Starting transcription... (Duration: {duration:.1f}s)",
            "duration": duration,
            "current_time": 0,
        }

        # Transcribe with word-level timestamps and progress tracking
        segments, info = whisper_model.transcribe(str(video_path), word_timestamps=True)

        processing_status[task_id] = {
            "status": "processing_segments",
            "progress": 25,
            "details": f"Processing audio segments... (Language: {info.language})",
            "language": info.language,
            "duration": duration,
        }

        # Write clean transcript (no header, just timestamps)
        with open(output_path, "w", encoding="utf-8") as f:

            line_buffer = ""
            current_time = 0
            segment_count = 0
            total_segments = 0

            # Count total segments first
            segments_list = list(segments)
            total_segments = len(segments_list)

            for segment_idx, segment in enumerate(segments_list):
                segment_count += 1

                # Update progress based on segments processed
                segment_progress = min(70, 25 + int((segment_count / total_segments) * 45))

                processing_status[task_id] = {
                    "status": "writing_transcript",
                    "progress": segment_progress,
                    "details": f"Segment {segment_count}/{total_segments} | Time: {format_timestamp(segment.start)} | Text: {segment.text[:50]}...",
                    "current_segment": segment_count,
                    "total_segments": total_segments,
                    "current_time": segment.start,
                    "duration": duration,
                }

                for word in segment.words:
                    if not line_buffer:
                        f.write(f"{format_timestamp(word.start)} ")
                        current_time = word.start

                    line_buffer += word.word + " "

                    # Write line every ~10 seconds or at sentence end
                    if word.end - current_time > 10.0 or word.word.strip().endswith((".", "!", "?")):
                        f.write(line_buffer.strip() + "\n")
                        line_buffer = ""

            # Write any remaining buffer
            if line_buffer.strip():
                f.write(line_buffer.strip() + "\n")

        processing_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "details": f"Transcription completed! {segment_count} segments processed ({duration:.1f}s audio)",
            "total_segments": segment_count,
            "duration": duration,
        }

    except Exception as e:
        processing_status[task_id] = {"status": "error", "message": f"Transcription error: {str(e)}"}


def extract_transcript_content_webapp(file_content):
    """
    Extracts just the transcript content, removing any existing title/summary header.
    Returns the clean transcript content for summarization.
    """
    lines = file_content.strip().split("\n")

    # If file starts with markdown header, find where transcript starts
    if file_content.startswith("# ") or file_content.startswith("TITLE:"):
        # Look for the separator line or first timestamp
        transcript_start = 0
        for i, line in enumerate(lines):
            if line.strip() == "---" or line.strip().startswith("["):
                transcript_start = i + 1 if line.strip() == "---" else i
                break

        # Return content from transcript start
        return "\n".join(lines[transcript_start:]).strip()

    # If no header found, return original content
    return file_content.strip()


def summarize_transcript_file(transcript_path, task_id, force_regenerate=False):
    """Add AI-generated title and summary to transcript"""
    try:
        processing_status[task_id] = {"status": "loading_transcript", "progress": 20}
        print(f"DEBUG: Starting summarization for {transcript_path}, force={force_regenerate}")

        # Load environment variables
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            print("DEBUG: No API key found")
            processing_status[task_id] = {"status": "error", "message": "OPENROUTER_API_KEY not found in .env file"}
            return

        # Read transcript content
        with open(transcript_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Check if already has title/summary
        has_header = original_content.startswith("TITLE:") or original_content.startswith("# ")

        if has_header and not force_regenerate:
            print(f"DEBUG: File has header and force={force_regenerate}, skipping")
            processing_status[task_id] = {
                "status": "completed",
                "progress": 100,
                "message": "Already has title/summary. Use force to regenerate.",
            }
            return

        # Extract clean transcript content for summarization
        if has_header:
            content = extract_transcript_content_webapp(original_content)
        else:
            content = original_content

        if not content:
            print("DEBUG: No content found")
            processing_status[task_id] = {"status": "error", "message": "No transcript content found"}
            return

        print(f"DEBUG: Content length: {len(content)}")
        processing_status[task_id] = {"status": "generating_summary", "progress": 50}

        # Generate summary using OpenRouter
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/video-app",
            "X-Title": "Video App Transcript Summarizer",
        }

        prompt = f"""Please analyze the following transcript and provide:
1. A concise, descriptive title (max 80 characters)
2. A brief summary (2-3 sentences) highlighting the main topics and key points

Transcript:
{content}

Please format your response as:
TITLE: [your title here]
SUMMARY: [your summary here]"""

        data = {
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.7,
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30
        )
        response.raise_for_status()

        result = response.json()
        ai_content = result["choices"][0]["message"]["content"]

        # Parse response
        lines = ai_content.strip().split("\n")
        title = ""
        summary = ""

        for line in lines:
            if line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()

        if title and summary:
            processing_status[task_id] = {"status": "writing_summary", "progress": 90}

            # Create header
            header = f"""# {title}

## Summary
{summary}

---

"""

            # Write updated content with clean transcript
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(header + content)

            processing_status[task_id] = {"status": "completed", "progress": 100, "title": title, "summary": summary}
        else:
            processing_status[task_id] = {"status": "error", "message": "Failed to parse AI response"}

    except Exception as e:
        processing_status[task_id] = {"status": "error", "message": str(e)}


def trim_video_file(input_path, output_path, start_time, end_time, task_id):
    """Trim video file using ffmpeg"""
    try:
        processing_status[task_id] = {"status": "trimming", "progress": 30}

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            "-y",
            output_path,
        ]

        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            processing_status[task_id] = {"status": "completed", "progress": 100}
        else:
            processing_status[task_id] = {"status": "error", "message": f"FFmpeg error: {result.stderr}"}

    except Exception as e:
        processing_status[task_id] = {"status": "error", "message": str(e)}


def compress_video_file(input_path, output_path, target_percentage, task_id):
    """Compress video file using ffmpeg with detailed progress tracking"""
    print(f"[COMPRESS] Starting compression for task {task_id}")
    print(f"[COMPRESS] Input: {input_path}")
    print(f"[COMPRESS] Output: {output_path}")
    print(f"[COMPRESS] Target: {target_percentage}%")

    try:
        processing_status[task_id] = {"status": "analyzing", "progress": 5, "details": "Getting video information..."}
        print(f"[COMPRESS] Status updated: analyzing")

        # Get video information using ffprobe
        probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", input_path]
        print(f"[COMPRESS] Running ffprobe command: {' '.join(probe_cmd)}")

        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[COMPRESS] ERROR: ffprobe failed with return code {result.returncode}")
            print(f"[COMPRESS] ffprobe stderr: {result.stderr}")
            processing_status[task_id] = {"status": "error", "message": "Failed to analyze video"}
            return

        print(f"[COMPRESS] ffprobe completed successfully")
        probe_data = json.loads(result.stdout)
        print(f"[COMPRESS] Probe data streams count: {len(probe_data.get('streams', []))}")

        video_stream = next((s for s in probe_data["streams"] if s["codec_type"] == "video"), None)
        if not video_stream:
            print(f"[COMPRESS] ERROR: No video stream found in probe data")
            processing_status[task_id] = {"status": "error", "message": "No video stream found"}
            return

        duration = float(probe_data["format"]["duration"])
        fps = eval(video_stream["r_frame_rate"])  # Convert fraction to float
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        total_frames = int(duration * fps) if duration > 0 else 0

        print(f"[COMPRESS] Video info - Duration: {duration:.2f}s, FPS: {fps:.2f}, Resolution: {width}x{height}")
        print(f"[COMPRESS] Estimated total frames: {total_frames}")

        processing_status[task_id] = {
            "status": "compressing",
            "progress": 10,
            "details": f"Starting compression ({width}x{height}, {fps:.1f}fps, ~{total_frames} frames)",
            "total_frames": total_frames,
            "current_frame": 0,
        }
        print(f"[COMPRESS] Status updated: compressing (10% progress)")

        # Determine compression settings based on target percentage
        if target_percentage >= 50:
            crf = 28
            scale_factor = 1.0
            preset = "medium"
        elif target_percentage >= 25:
            crf = 32
            scale_factor = 0.8
            preset = "medium"
        else:
            crf = 35
            scale_factor = 0.5
            preset = "fast"  # Use faster preset for heavy compression

        print(f"[COMPRESS] Compression settings - CRF: {crf}, Scale: {scale_factor}, Preset: {preset}")

        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Ensure dimensions are even (required by some codecs)
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)

        print(f"[COMPRESS] Target resolution: {new_width}x{new_height} (scale factor: {scale_factor})")

        # Build ffmpeg command with progress output
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-vf",
            f"scale={new_width}:{new_height}",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-progress",
            "pipe:1",  # Output progress to stdout
            "-y",
            output_path,
        ]

        print(f"[COMPRESS] FFmpeg command: {' '.join(cmd)}")

        # Run ffmpeg with real-time progress tracking
        print(f"[COMPRESS] Starting FFmpeg process")
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, bufsize=1
        )
        print(f"[COMPRESS] FFmpeg process started with PID: {process.pid}")

        # Parse progress output in real-time
        current_frame = 0
        current_fps = 0
        current_bitrate = "N/A"
        current_speed = "N/A"
        progress_update_count = 0

        print(f"[COMPRESS] Starting progress monitoring")

        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                print(f"[COMPRESS] FFmpeg process finished, no more output")
                break

            if output:
                line = output.strip()
                if line.startswith("frame="):
                    try:
                        prev_frame = current_frame
                        current_frame = int(line.split("=")[1].strip())
                        if current_frame != prev_frame and current_frame % 100 == 0:  # Log every 100 frames
                            print(f"[COMPRESS] Progress: frame {current_frame}")
                    except (ValueError, IndexError):
                        print(f"[COMPRESS] WARNING: Could not parse frame from: {line}")
                elif line.startswith("fps="):
                    try:
                        current_fps = float(line.split("=")[1].strip())
                    except (ValueError, IndexError):
                        print(f"[COMPRESS] WARNING: Could not parse fps from: {line}")
                elif line.startswith("bitrate="):
                    try:
                        current_bitrate = line.split("=")[1].strip()
                    except (ValueError, IndexError):
                        print(f"[COMPRESS] WARNING: Could not parse bitrate from: {line}")
                elif line.startswith("speed="):
                    try:
                        current_speed = line.split("=")[1].strip()
                    except (ValueError, IndexError):
                        print(f"[COMPRESS] WARNING: Could not parse speed from: {line}")

                # Update progress based on frames processed
                if total_frames > 0 and current_frame > 0:
                    frame_progress = min(85, int((current_frame / total_frames) * 85))
                    progress = max(10, frame_progress)  # Ensure progress doesn't go backwards

                    processing_status[task_id] = {
                        "status": "compressing",
                        "progress": progress,
                        "details": f"Frame {current_frame:,}/{total_frames:,} | {current_fps:.1f}fps | {current_bitrate} | {current_speed}x",
                        "total_frames": total_frames,
                        "current_frame": current_frame,
                        "fps": current_fps,
                        "bitrate": current_bitrate,
                        "speed": current_speed,
                    }

                    progress_update_count += 1
                    if progress_update_count % 500 == 0:  # Log every 500 updates
                        print(f"[COMPRESS] Status update #{progress_update_count}: {progress}% complete")

        # Wait for process to complete
        print(f"[COMPRESS] Waiting for FFmpeg process to complete")
        stderr_output = process.stderr.read()
        return_code = process.wait()

        print(f"[COMPRESS] FFmpeg process completed with return code: {return_code}")
        if stderr_output:
            print(f"[COMPRESS] FFmpeg stderr output: {stderr_output[:500]}...")  # First 500 chars

        if return_code == 0:
            # Check if output file was created and get its size
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                input_size = os.path.getsize(input_path)
                compression_ratio = (output_size / input_size) * 100 if input_size > 0 else 0
                print(f"[COMPRESS] SUCCESS: Output file created")
                print(f"[COMPRESS] Input size: {format_file_size(input_size)}")
                print(f"[COMPRESS] Output size: {format_file_size(output_size)}")
                print(f"[COMPRESS] Actual compression ratio: {compression_ratio:.1f}%")

                processing_status[task_id] = {
                    "status": "completed",
                    "progress": 100,
                    "details": f"Compression completed! Final: {current_frame:,} frames processed. Size: {format_file_size(input_size)} → {format_file_size(output_size)} ({compression_ratio:.1f}%)",
                    "total_frames": total_frames,
                    "current_frame": current_frame,
                    "input_size": input_size,
                    "output_size": output_size,
                    "compression_ratio": compression_ratio,
                }
            else:
                print(f"[COMPRESS] ERROR: Output file was not created despite return code 0")
                processing_status[task_id] = {"status": "error", "message": "Output file was not created"}
        else:
            print(f"[COMPRESS] ERROR: FFmpeg failed with return code {return_code}")
            # Parse stderr for more detailed error information
            error_msg = "FFmpeg compression failed"
            if stderr_output:
                print(f"[COMPRESS] Analyzing stderr for error details")
                # Extract meaningful error from stderr
                lines = stderr_output.strip().split("\n")
                for line in reversed(lines):
                    if "error" in line.lower() or "failed" in line.lower():
                        error_msg = line.strip()
                        print(f"[COMPRESS] Found error in stderr: {error_msg}")
                        break
                if error_msg == "FFmpeg compression failed":
                    error_msg = f"FFmpeg error (code {return_code}): {stderr_output[-200:]}"  # Last 200 chars
                    print(f"[COMPRESS] Using generic error message: {error_msg}")

            processing_status[task_id] = {"status": "error", "message": error_msg}

    except Exception as e:
        print(f"[COMPRESS] EXCEPTION: Unexpected error during compression: {str(e)}")
        print(f"[COMPRESS] Exception type: {type(e).__name__}")
        import traceback

        print(f"[COMPRESS] Traceback: {traceback.format_exc()}")
        processing_status[task_id] = {"status": "error", "message": f"Compression error: {str(e)}"}

    print(f"[COMPRESS] Compression function completed for task {task_id}")
    final_status = processing_status.get(task_id, {"status": "unknown"})
    print(f"[COMPRESS] Final status: {final_status.get('status', 'unknown')}")


def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"


# Video Processing API Routes
@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    """Start transcription for one or more videos"""
    data = request.get_json()
    if not data or "videos" not in data:
        return jsonify({"error": "No videos specified"}), 400

    video_dir = get_video_directory()
    results = []

    for video_filename in data["videos"]:
        video_path = os.path.join(video_dir, video_filename)
        if not os.path.exists(video_path):
            results.append({"video": video_filename, "error": "Video not found"})
            continue

        video_name = os.path.splitext(video_filename)[0]
        transcript_path = os.path.join(video_dir, f"{video_name}_transcript.md")

        # Check if transcript already exists
        if os.path.exists(transcript_path):
            results.append({"video": video_filename, "status": "exists", "message": "Transcript already exists"})
            continue

        # Generate unique task ID
        task_id = f"transcribe_{video_name}_{int(time.time())}"
        processing_status[task_id] = {"status": "queued", "progress": 0}

        # Start transcription in background
        thread = threading.Thread(target=transcribe_video_file, args=(video_path, transcript_path, task_id))
        thread.daemon = True
        thread.start()

        results.append({"video": video_filename, "task_id": task_id, "status": "started"})

    return jsonify({"results": results})


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    """Start summarization for one or more transcripts"""
    data = request.get_json()
    if not data or "videos" not in data:
        return jsonify({"error": "No videos specified"}), 400

    video_dir = get_video_directory()
    results = []

    for video_filename in data["videos"]:
        video_name = os.path.splitext(video_filename)[0]
        transcript_path = os.path.join(video_dir, f"{video_name}_transcript.md")

        if not os.path.exists(transcript_path):
            results.append({"video": video_filename, "error": "Transcript not found"})
            continue

        # Generate unique task ID
        task_id = f"summarize_{video_name}_{int(time.time())}"
        processing_status[task_id] = {"status": "queued", "progress": 0}

        # Start summarization in background
        force_regenerate = data.get("force", False)
        thread = threading.Thread(target=summarize_transcript_file, args=(transcript_path, task_id, force_regenerate))
        thread.daemon = True
        thread.start()

        results.append({"video": video_filename, "task_id": task_id, "status": "started"})

    return jsonify({"results": results})


@app.route("/api/trim", methods=["POST"])
def api_trim():
    """Trim a video file"""
    data = request.get_json()
    required_fields = ["video", "start_time", "end_time"]

    if not data or not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields: video, start_time, end_time"}), 400

    video_dir = get_video_directory()
    video_path = os.path.join(video_dir, data["video"])

    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404

    # Generate output filename
    video_name = os.path.splitext(data["video"])[0]
    video_ext = os.path.splitext(data["video"])[1]
    output_filename = f"{video_name}_trimmed{video_ext}"
    output_path = os.path.join(video_dir, output_filename)

    # Generate unique task ID
    task_id = f"trim_{video_name}_{int(time.time())}"
    processing_status[task_id] = {"status": "queued", "progress": 0}

    # Start trimming in background
    thread = threading.Thread(
        target=trim_video_file, args=(video_path, output_path, data["start_time"], data["end_time"], task_id)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"task_id": task_id, "output_filename": output_filename, "status": "started"})


@app.route("/api/compress", methods=["POST"])
def api_compress():
    """Compress a video file"""
    print("[COMPRESS API] Starting compression request")

    data = request.get_json()
    print(f"[COMPRESS API] Request data: {data}")

    required_fields = ["video", "target_percentage"]

    if not data or not all(field in data for field in required_fields):
        print(f"[COMPRESS API] ERROR: Missing required fields. Data: {data}")
        return jsonify({"error": "Missing required fields: video, target_percentage"}), 400

    video_dir = get_video_directory()
    video_path = os.path.join(video_dir, data["video"])
    print(f"[COMPRESS API] Video directory: {video_dir}")
    print(f"[COMPRESS API] Video path: {video_path}")

    if not os.path.exists(video_path):
        print(f"[COMPRESS API] ERROR: Video file not found at {video_path}")
        return jsonify({"error": "Video not found"}), 404

    # Get file size for logging
    file_size = os.path.getsize(video_path)
    print(f"[COMPRESS API] Input file size: {format_file_size(file_size)}")

    # Validate percentage
    try:
        target_percentage = int(data["target_percentage"])
        if target_percentage < 10 or target_percentage > 90:
            print(f"[COMPRESS API] ERROR: Invalid target percentage {target_percentage}")
            return jsonify({"error": "Target percentage must be between 10 and 90"}), 400
        print(f"[COMPRESS API] Target compression: {target_percentage}%")
    except ValueError:
        print(f"[COMPRESS API] ERROR: Invalid target percentage format: {data['target_percentage']}")
        return jsonify({"error": "Invalid target percentage"}), 400

    # Generate output filename
    video_name = os.path.splitext(data["video"])[0]
    video_ext = os.path.splitext(data["video"])[1]
    output_filename = f"{video_name}_{target_percentage}percent{video_ext}"
    output_path = os.path.join(video_dir, output_filename)
    print(f"[COMPRESS API] Output filename: {output_filename}")
    print(f"[COMPRESS API] Output path: {output_path}")

    # Check if output file already exists
    if os.path.exists(output_path):
        print(f"[COMPRESS API] WARNING: Output file already exists, will overwrite")

    # Generate unique task ID
    task_id = f"compress_{video_name}_{int(time.time())}"
    processing_status[task_id] = {"status": "queued", "progress": 0}
    print(f"[COMPRESS API] Generated task ID: {task_id}")

    # Start compression in background
    print(f"[COMPRESS API] Starting background compression thread")
    thread = threading.Thread(target=compress_video_file, args=(video_path, output_path, target_percentage, task_id))
    thread.daemon = True
    thread.start()

    print(f"[COMPRESS API] Compression started successfully for task {task_id}")
    return jsonify({"task_id": task_id, "output_filename": output_filename, "status": "started"})


@app.route("/api/status/<task_id>")
def api_status(task_id):
    """Get status of a processing task"""
    if task_id not in processing_status:
        print(f"[STATUS API] Task not found: {task_id}")
        print(f"[STATUS API] Available tasks: {list(processing_status.keys())}")
        return jsonify({"error": "Task not found"}), 404

    status = processing_status[task_id]
    if task_id.startswith("compress_"):
        print(
            f"[STATUS API] Compression task {task_id} status: {status.get('status', 'unknown')} ({status.get('progress', 0)}%)"
        )

    return jsonify(status)


@app.route("/api/replace", methods=["POST"])
def api_replace():
    """Replace original video with processed version"""
    data = request.get_json()
    if not data or "filename" not in data or "processed_filename" not in data:
        return jsonify({"error": "Missing filename or processed_filename"}), 400

    original_path = os.path.join(VIDEO_DIR, data["filename"])
    processed_path = os.path.join(VIDEO_DIR, data["processed_filename"])

    if not os.path.exists(processed_path):
        return jsonify({"error": "Processed file not found"}), 404

    try:
        # Create backup of original
        backup_path = original_path + ".backup"
        shutil.copy2(original_path, backup_path)

        # Replace original with processed
        shutil.move(processed_path, original_path)

        # Clear cache to refresh metadata
        cache_file = os.path.join(VIDEO_DIR, ".video_metadata_cache.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)

        return jsonify({"success": True, "backup_created": backup_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete", methods=["POST"])
def api_delete():
    """Delete video and associated transcript"""
    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "Missing filename"}), 400

    filename = data["filename"]
    video_path = os.path.join(VIDEO_DIR, filename)

    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    try:
        deleted_files = []

        # Delete video file
        os.remove(video_path)
        deleted_files.append(filename)

        # Delete associated transcript if it exists
        video_name = os.path.splitext(filename)[0]
        transcript_path = os.path.join(VIDEO_DIR, f"{video_name}_transcript.md")
        if os.path.exists(transcript_path):
            os.remove(transcript_path)
            deleted_files.append(f"{video_name}_transcript.md")

        # Delete any backup files
        backup_path = video_path + ".backup"
        if os.path.exists(backup_path):
            os.remove(backup_path)
            deleted_files.append(f"{filename}.backup")

        # Clear cache to refresh video list
        cache_file = os.path.join(VIDEO_DIR, ".video_metadata_cache.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)

        return jsonify({"success": True, "deleted_files": deleted_files})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete_bulk", methods=["POST"])
def api_delete_bulk():
    """Delete multiple videos and their transcripts"""
    data = request.get_json()
    if not data or "filenames" not in data:
        return jsonify({"error": "Missing filenames"}), 400

    filenames = data["filenames"]
    if not isinstance(filenames, list) or not filenames:
        return jsonify({"error": "Invalid filenames list"}), 400

    deleted_files = []
    errors = []

    for filename in filenames:
        video_path = os.path.join(VIDEO_DIR, filename)

        if not os.path.exists(video_path):
            errors.append(f"Video not found: {filename}")
            continue

        try:
            # Delete video file
            os.remove(video_path)
            deleted_files.append(filename)

            # Delete associated transcript if it exists
            video_name = os.path.splitext(filename)[0]
            transcript_path = os.path.join(VIDEO_DIR, f"{video_name}_transcript.md")
            if os.path.exists(transcript_path):
                os.remove(transcript_path)
                deleted_files.append(f"{video_name}_transcript.md")

            # Delete any backup files
            backup_path = video_path + ".backup"
            if os.path.exists(backup_path):
                os.remove(backup_path)
                deleted_files.append(f"{filename}.backup")

        except Exception as e:
            errors.append(f"Error deleting {filename}: {str(e)}")

    # Clear cache to refresh video list
    cache_file = os.path.join(VIDEO_DIR, ".video_metadata_cache.json")
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
        except:
            pass

    return jsonify({"success": len(deleted_files) > 0, "deleted_files": deleted_files, "errors": errors})


@app.route("/")
def index():
    """Main page showing all videos"""
    videos = get_all_videos()
    for video in videos:
        video["formatted_size"] = format_file_size(video["size"])
        video["formatted_duration"] = format_duration(video["duration"])
    return render_template("index.html", videos=videos)


@app.route("/video/<path:filename>")
def video_player(filename):
    """Video player page with transcript"""
    video_dir = get_video_directory()
    video_path = os.path.join(video_dir, filename)

    if not os.path.exists(video_path):
        return "Video not found", 404

    # Load metadata cache for this single video
    metadata_cache = load_metadata_cache()
    video_info = get_video_info(video_path, metadata_cache)

    # Save cache if it was updated
    save_metadata_cache(metadata_cache)

    return render_template("player.html", video=video_info)


@app.route("/serve/<path:filename>")
def serve_video(filename):
    """Serve video file"""
    video_dir = get_video_directory()
    video_path = os.path.join(video_dir, filename)

    if not os.path.exists(video_path):
        return "File not found", 404

    return send_file(video_path)


# Template filters
@app.template_filter("truncate_text")
def truncate_text(text, length=100):
    """Truncate text to specified length"""
    if not text:
        return ""
    return text[:length] + "..." if len(text) > length else text


@app.template_filter("timestamp_to_date")
def timestamp_to_date(timestamp):
    """Convert timestamp to readable date"""
    import datetime

    try:
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return "Unknown"


@app.route("/transcript/<path:filename>", methods=["GET", "POST"])
def transcript_editor(filename):
    """Handle transcript editing - GET to load, POST to save"""
    video_dir = get_video_directory()
    video_name = os.path.splitext(filename)[0]
    transcript_path = os.path.join(video_dir, f"{video_name}_transcript.md")

    if request.method == "GET":
        # Load transcript content
        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return content
            except Exception as e:
                return f"Error reading transcript: {str(e)}", 500
        else:
            # Return template for new transcript
            return f"# {video_name}\n\n## Summary\n\n## Transcript\n\n[00:00:00] "

    elif request.method == "POST":
        # Save transcript content
        try:
            data = request.get_json()
            if not data or "content" not in data:
                return jsonify({"success": False, "error": "No content provided"})

            content = data["content"]

            # Ensure the directory exists
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)

            # Write the transcript file
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(content)

            return jsonify({"success": True, "message": "Transcript saved successfully"})

        except Exception as e:
            return jsonify({"success": False, "error": str(e)})


# Face Recognition API Routes
@app.route("/api/face_recognition", methods=["POST"])
def api_face_recognition():
    """Start face recognition for one or more videos"""
    data = request.get_json()
    if not data or "videos" not in data:
        return jsonify({"error": "No videos specified"}), 400

    video_dir = get_video_directory()
    results = []

    for video_filename in data["videos"]:
        video_path = os.path.join(video_dir, video_filename)
        if not os.path.exists(video_path):
            results.append({"video": video_filename, "error": "Video not found"})
            continue

        video_name = os.path.splitext(video_filename)[0]
        faces_path = os.path.join(video_dir, f"{video_name}_faces.json")

        # Check if face recognition already exists
        force_regenerate = data.get("force", False)
        if os.path.exists(faces_path) and not force_regenerate:
            results.append(
                {"video": video_filename, "status": "exists", "message": "Face recognition data already exists"}
            )
            continue

        # Generate unique task ID
        task_id = f"faces_{video_name}_{int(time.time())}"
        processing_status[task_id] = {"status": "queued", "progress": 0}

        # Start face recognition in background
        thread = threading.Thread(target=process_video_face_recognition, args=(video_path, task_id, processing_status))
        thread.daemon = True
        thread.start()

        results.append({"video": video_filename, "task_id": task_id, "status": "started"})

    return jsonify({"results": results})


@app.route("/api/combined_processing", methods=["POST"])
def api_combined_processing():
    """Start both transcript and face recognition processing for videos"""
    data = request.get_json()
    if not data or "videos" not in data:
        return jsonify({"error": "No videos specified"}), 400

    video_dir = get_video_directory()
    results = []

    for video_filename in data["videos"]:
        video_path = os.path.join(video_dir, video_filename)
        if not os.path.exists(video_path):
            results.append({"video": video_filename, "error": "Video not found"})
            continue

        video_name = os.path.splitext(video_filename)[0]
        transcript_path = os.path.join(video_dir, f"{video_name}_transcript.md")
        faces_path = os.path.join(video_dir, f"{video_name}_faces.json")

        force_regenerate = data.get("force", False)
        video_result = {"video": video_filename, "tasks": []}

        # Start transcription if needed
        if not os.path.exists(transcript_path) or force_regenerate:
            transcript_task_id = f"transcribe_{video_name}_{int(time.time())}"
            processing_status[transcript_task_id] = {"status": "queued", "progress": 0}

            thread = threading.Thread(
                target=transcribe_video_file, args=(video_path, transcript_path, transcript_task_id)
            )
            thread.daemon = True
            thread.start()

            video_result["tasks"].append({"type": "transcript", "task_id": transcript_task_id, "status": "started"})

        # Start face recognition if needed
        if not os.path.exists(faces_path) or force_regenerate:
            faces_task_id = f"faces_{video_name}_{int(time.time())}"
            processing_status[faces_task_id] = {"status": "queued", "progress": 0}

            thread = threading.Thread(
                target=process_video_face_recognition, args=(video_path, faces_task_id, processing_status)
            )
            thread.daemon = True
            thread.start()

            video_result["tasks"].append({"type": "faces", "task_id": faces_task_id, "status": "started"})

        if not video_result["tasks"]:
            video_result["message"] = "Both transcript and face recognition data already exist"

        results.append(video_result)

    return jsonify({"results": results})


@app.route("/face_library")
def face_library_page():
    """Face library management page"""
    face_lib = FaceLibrary()
    persons = face_lib.get_all_persons()

    library_info = []
    for person in persons:
        library_info.append({"name": person, "encoding_count": face_lib.get_person_count(person)})

    return render_template("face_library.html", persons=library_info)


@app.route("/api/face_library", methods=["GET", "POST", "DELETE"])
def api_face_library():
    """Face library management API"""
    face_lib = FaceLibrary()

    if request.method == "GET":
        # Get all persons in library
        persons = face_lib.get_all_persons()
        library_info = []
        for person in persons:
            library_info.append({"name": person, "encoding_count": face_lib.get_person_count(person)})
        return jsonify({"persons": library_info})

    elif request.method == "POST":
        data = request.get_json()
        action = data.get("action")

        if action == "rename":
            old_name = data.get("old_name")
            new_name = data.get("new_name")
            if face_lib.rename_person(old_name, new_name):
                return jsonify({"success": True, "message": f"Renamed {old_name} to {new_name}"})
            else:
                return jsonify({"success": False, "error": "Failed to rename person"}), 400

        elif action == "build_from_photos":
            # Build library from photos in face_library/photos folder
            def progress_callback(message, progress):
                # Could implement WebSocket for real-time updates
                pass

            results = face_lib.build_from_photos(progress_callback)
            return jsonify(
                {"success": True, "message": f"Added {results['added']} face encodings", "errors": results["errors"]}
            )

        else:
            return jsonify({"error": "Invalid action"}), 400

    elif request.method == "DELETE":
        data = request.get_json()
        person_name = data.get("name")
        if face_lib.remove_person(person_name):
            return jsonify({"success": True, "message": f"Removed {person_name} from library"})
        else:
            return jsonify({"success": False, "error": "Failed to remove person"}), 400


@app.route("/faces/<path:filename>", methods=["GET", "POST"])
def face_data_editor(filename):
    """Handle face recognition data viewing and editing"""
    video_dir = get_video_directory()
    video_name = os.path.splitext(filename)[0]
    faces_path = os.path.join(video_dir, f"{video_name}_faces.json")

    if request.method == "GET":
        # Load face recognition data
        if os.path.exists(faces_path):
            try:
                with open(faces_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return content
            except Exception as e:
                return f"Error reading face data: {str(e)}", 500
        else:
            return jsonify({"error": "Face recognition data not found"}), 404

    elif request.method == "POST":
        # Save updated face recognition data
        try:
            data = request.get_json()
            if not data or "content" not in data:
                return jsonify({"success": False, "error": "No content provided"})

            content = data["content"]

            # Validate JSON
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                return jsonify({"success": False, "error": f"Invalid JSON: {str(e)}"})

            # Write the face data file
            with open(faces_path, "w", encoding="utf-8") as f:
                f.write(content)

            return jsonify({"success": True, "message": "Face data saved successfully"})

        except Exception as e:
            return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":

    print("📹 Video Viewer Server Starting...")
    print(f"📁 Video Directory: {get_video_directory()}")
    print("🌐 Open http://localhost:8008 in your browser")
    print("\n⚙️  To configure:")
    print("   1. Update the VIDEO_DIR variable in the script")
    print("   2. Make sure your video files are in that directory")
    print("   3. Transcript files should be named: {video_name}_transcript.md")

    app.run(debug=True, host="0.0.0.0", port=8008)
