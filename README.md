# Video Library with Integrated Processing

A comprehensive video management application with built-in transcription, summarization, trimming, and compression capabilities.

## Features

### 📹 Video Management
- **Library View**: Browse videos in card or table view
- **Metadata Display**: Shows file size, duration, creation date
- **Smart Sorting**: Sort by title, filename, duration, size, date, or transcript status
- **Search & Filter**: Find videos quickly with various filters

### 🎤 Transcription & AI Features
- **Automatic Transcription**: Uses Faster Whisper for accurate speech-to-text
- **AI Summarization**: Generates titles and summaries using OpenRouter API
- **Bulk Processing**: Select multiple videos for batch transcription/summarization
- **Real-time Progress**: Track processing status with progress bars
- **Transcript Editing**: Built-in editor with markdown support

### ✂️ Video Processing
- **Video Trimming**: Cut videos by time range with precision controls
- **Video Compression**: Reduce file size with preset or custom compression ratios
- **Preview System**: Preview processed videos before replacing originals
- **Backup Protection**: Original files are automatically backed up when replaced

### 🎬 Player Features
- **Integrated Player**: HTML5 video player with transcript synchronization
- **Clickable Timestamps**: Jump to specific moments in the video
- **Search Transcripts**: Find and highlight text within transcripts
- **Processing Controls**: Trim and compress directly from the player

## Installation

1. **Install Dependencies**:
   ```bash
   poetry install
   ```

2. **Set up Environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key
   ```

3. **Configure Video Directory**:
   Edit `video_viewer_app.py` and update the `VIDEO_DIR` variable:
   ```python
   VIDEO_DIR = "/path/to/your/videos"
   ```

4. **Install System Dependencies**:
   - **FFmpeg**: Required for video processing
     ```bash
     # Ubuntu/Debian
     sudo apt install ffmpeg
     
     # macOS
     brew install ffmpeg
     ```
   - **CUDA** (optional): For GPU-accelerated transcription

## Usage

### Starting the Application
```bash
python video_viewer_app.py
```
Access the application at `http://localhost:5000`

### Video Processing Workflow

#### Individual Video Processing (Player View)
1. Click on any video to open the player
2. Use the **Video Processing** panel for:
   - **Trimming**: Set start/end times and click "Trim"
   - **Compression**: Choose preset (50%, 25%, 10%) or custom percentage
3. Monitor progress in real-time
4. Preview the result before replacing the original

#### Bulk Processing (Library View)
1. Select videos using checkboxes
2. Choose bulk action:
   - **🎤 Transcribe Selected**: Generate transcripts for multiple videos
   - **📝 Summarize Selected**: Add AI-generated titles and summaries
3. Track progress for all selected videos
4. Refresh page when complete to see updates

### Transcript Management
- **Auto-generation**: Transcripts are created with timestamps every ~10 seconds
- **Manual Editing**: Click "✏️ Edit" to modify transcripts directly
- **Format Support**: Markdown format with title, summary, and timestamped content
- **Search**: Find specific text within transcripts

## API Endpoints

The application provides REST API endpoints for programmatic access:

- `POST /api/transcribe` - Start transcription for videos
- `POST /api/summarize` - Generate summaries for transcripts
- `POST /api/trim` - Trim video files
- `POST /api/compress` - Compress video files
- `GET /api/status/<task_id>` - Check processing status
- `POST /api/replace` - Replace original with processed video

## Configuration

### Environment Variables
```bash
# Required for summarization
OPENROUTER_API_KEY=your_api_key_here
```

### Video Directory Structure
```
/your/video/directory/
├── video1.mp4
├── video1_transcript.md
├── video2.mkv
├── video2_transcript.md
└── .video_metadata_cache.json
```

### Supported Formats
- **Video**: MP4, AVI, MOV, MKV, WEBM
- **Transcripts**: Markdown (.md) files

## Technical Details

### Architecture
- **Backend**: Flask web application
- **Frontend**: HTML/CSS/JavaScript with responsive design
- **Processing**: Background threads for long-running operations
- **Storage**: File-based with JSON metadata caching

### Processing Pipeline
1. **Transcription**: Faster Whisper → Timestamped text
2. **Summarization**: OpenRouter API → Title + Summary
3. **Video Processing**: FFmpeg → Trimmed/Compressed output
4. **Status Tracking**: Real-time progress updates

### Performance Optimizations
- **Metadata Caching**: Avoids re-analyzing unchanged videos
- **Background Processing**: Non-blocking operations
- **Progress Tracking**: Real-time status updates
- **Smart Date Extraction**: Parses creation dates from filenames

## Troubleshooting

### Common Issues
1. **Transcription fails**: Check CUDA/CPU setup and model availability
2. **Summarization fails**: Verify OpenRouter API key in `.env`
3. **Video processing fails**: Ensure FFmpeg is installed and accessible
4. **No videos shown**: Check `VIDEO_DIR` path in configuration

### System Requirements
- **Python**: 3.12+
- **Memory**: 4GB+ RAM (8GB+ recommended for large videos)
- **Storage**: Space for original + processed videos
- **GPU**: Optional CUDA-compatible GPU for faster transcription

## Development

### Project Structure
```
video_app/
├── video_viewer_app.py      # Main Flask application
├── video_transcriber.py     # Transcription functionality
├── transcript_summarizer.py # AI summarization
├── trim.sh                  # Video trimming script
├── compress.sh              # Video compression script
├── templates/
│   ├── index.html          # Library view
│   └── player.html         # Video player
├── pyproject.toml          # Dependencies
└── .env.example           # Environment template
```

### Adding New Features
1. **Backend**: Add routes in `video_viewer_app.py`
2. **Frontend**: Update templates with new UI components
3. **Processing**: Create background functions for long operations
4. **API**: Follow existing patterns for status tracking

## License

This project integrates multiple video processing tools into a unified web interface for efficient video management and processing workflows.
