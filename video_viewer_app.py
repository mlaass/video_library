#!/usr/bin/env python3
"""
Video Viewer Flask Application
A simple web server to view videos with transcript integration
"""

import os
import re
from pathlib import Path
from flask import Flask, render_template, send_file, url_for, request
import json
import ffmpeg

app = Flask(__name__)

# Configuration
VIDEO_DIR = "~/workspace/video_work/screen_recording"  # Update this path
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


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


def get_video_info(video_path):
    """Get video information including transcript data if available"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    transcript_path = os.path.join(os.path.dirname(video_path), f"{video_name}_transcript.md")

    info = {
        "filename": os.path.basename(video_path),
        "name": video_name,
        "path": video_path,
        "size": os.path.getsize(video_path),
        "modified_time": os.path.getmtime(video_path),
        "duration": get_video_duration(video_path),
        "has_transcript": os.path.exists(transcript_path),
    }

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

    return info


def get_all_videos():
    """Get all video files from the directory"""
    video_dir = get_video_directory()
    videos = []

    if not os.path.exists(video_dir):
        return videos

    for filename in os.listdir(video_dir):
        if any(filename.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
            video_path = os.path.join(video_dir, filename)
            video_info = get_video_info(video_path)
            videos.append(video_info)

    # Sort by modification time (newest first)
    videos.sort(key=lambda x: x["modified_time"], reverse=True)
    return videos


def get_video_duration(video_path):
    """Get video duration in seconds using ffmpeg"""
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['streams'][0]['duration'])
        return duration
    except Exception as e:
        print(f"Error getting duration for {video_path}: {e}")
        return None


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

    video_info = get_video_info(video_path)
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


if __name__ == "__main__":

    print("📹 Video Viewer Server Starting...")
    print(f"📁 Video Directory: {get_video_directory()}")
    print("🌐 Open http://localhost:5000 in your browser")
    print("\n⚙️  To configure:")
    print("   1. Update the VIDEO_DIR variable in the script")
    print("   2. Make sure your video files are in that directory")
    print("   3. Transcript files should be named: {video_name}_transcript.md")

    app.run(debug=True, host="0.0.0.0", port=5000)
