#!/usr/bin/env python3
"""
Video Viewer Flask Application
A simple web server to view videos with transcript integration
"""

import os
import re
from pathlib import Path
from flask import Flask, render_template, send_file, url_for, request, jsonify
import json
import ffmpeg

app = Flask(__name__)

# Configuration
VIDEO_DIR = "~/workspace/video_work/screen_recording"  # Update this path
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
METADATA_CACHE_FILE = ".video_metadata_cache.json"


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
    """Get video information including transcript data if available"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    transcript_path = os.path.join(os.path.dirname(video_path), f"{video_name}_transcript.md")

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
        with open(cache_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_metadata_cache(cache):
    """Save video metadata cache to JSON file"""
    video_dir = get_video_directory()
    cache_path = os.path.join(video_dir, METADATA_CACHE_FILE)
    
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata cache: {e}")


def get_video_duration(video_path):
    """Get video duration in seconds using ffmpeg"""
    try:
        probe = ffmpeg.probe(video_path)
        return float(probe['streams'][0]['duration'])
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
        if cached_data.get('mtime') == file_mtime:
            return cached_data.get('duration'), cached_data.get('video_date')
    
    # Extract duration and video date
    duration = get_video_duration(video_path)
    
    # Try to extract date from filename, fall back to file creation time
    video_date = extract_date_from_filename(filename)
    if video_date is None:
        # Try to get file birth time (creation time) if available
        try:
            stat = os.stat(video_path)
            # On Linux, try st_birthtime (not always available), fall back to st_ctime
            video_date = getattr(stat, 'st_birthtime', stat.st_ctime)
        except (AttributeError, OSError):
            # Fall back to modification time
            video_date = file_mtime
    
    # Update cache
    cache[filename] = {
        'duration': duration,
        'video_date': video_date,
        'mtime': file_mtime
    }
    
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
    match = re.match(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})', filename)
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
                with open(transcript_path, 'r', encoding='utf-8') as f:
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
            if not data or 'content' not in data:
                return jsonify({"success": False, "error": "No content provided"})
            
            content = data['content']
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
            
            # Write the transcript file
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return jsonify({"success": True, "message": "Transcript saved successfully"})
            
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":

    print("📹 Video Viewer Server Starting...")
    print(f"📁 Video Directory: {get_video_directory()}")
    print("🌐 Open http://localhost:5000 in your browser")
    print("\n⚙️  To configure:")
    print("   1. Update the VIDEO_DIR variable in the script")
    print("   2. Make sure your video files are in that directory")
    print("   3. Transcript files should be named: {video_name}_transcript.md")

    app.run(debug=True, host="0.0.0.0", port=5000)
