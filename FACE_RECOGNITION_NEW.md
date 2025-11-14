# Face Recognition System - SQLite-Based

## Overview

The face recognition system has been completely redesigned to use a **scan-first, label-later** workflow with SQLite database storage.

## Key Features

### 1. **Automatic Face Detection & Storage**
- When processing videos, all detected faces are automatically saved
- Each face gets a unique hash based on its encoding
- Face crops saved as `<hash>_<uuid>.jpg` in `face_library/faces/`
- No pre-built library required to start

### 2. **SQLite Database**
- **face_identities**: Unique face identities (one per person)
- **face_instances**: Individual face detections (many per identity)
- **face_operations**: History of merge/split operations

### 3. **Label-as-You-Go Workflow**
1. Process videos → faces detected automatically
2. View unlabeled faces in Face Library UI
3. Label faces with person names
4. Future detections automatically use labels

### 4. **Advanced Operations**

#### Labeling
- Select one or more face identities
- Assign a person name
- All instances of that face get the label

#### Merging
- Select multiple face identities that are the same person
- Merge them into one identity
- All instances consolidated under one hash
- Useful for fixing detection variations

#### Splitting
- Select specific instances from one identity
- Split them into a new identity
- Useful when multiple people were incorrectly grouped

## File Structure

```
video_library/
├── face_library/
│   ├── faces/                    # All face crops
│   │   ├── abc123_uuid1.jpg     # Face instance
│   │   ├── abc123_uuid2.jpg     # Same person, different frame
│   │   └── def456_uuid3.jpg     # Different person
│   └── face_library.db          # SQLite database
└── video_name_faces.json        # Per-video face data
```

## Database Schema

### face_identities
- `hash` (PRIMARY KEY): Unique face encoding hash
- `label`: Person name (NULL if unlabeled)
- `canonical_encoding`: Face encoding (pickled numpy array)
- `first_seen`, `last_seen`: Timestamps
- `appearance_count`: Number of instances

### face_instances
- `id` (PRIMARY KEY): Auto-increment ID
- `face_hash`: Links to face_identities
- `image_filename`: Face crop filename
- `video_filename`: Source video
- `timestamp`: Time in video (seconds)
- `detection_confidence`: Detection quality
- `created_at`: When detected

### face_operations
- Tracks merge/split history for auditing

## API Endpoints

### GET `/face_library`
Face library management page with filtering

Query params:
- `filter`: `all` | `labeled` | `unlabeled`

### GET `/api/face_library`
Get face identities

Query params:
- `filter`: `all` | `labeled` | `unlabeled`
- `hash`: Get specific identity with instances

### POST `/api/face_library`
Perform face operations

Actions:
```json
// Label a face
{
  "action": "label",
  "hash": "abc123...",
  "label": "John Doe"
}

// Merge faces
{
  "action": "merge",
  "source_hashes": ["abc123...", "def456..."],
  "target_hash": "abc123...",
  "notes": "Same person"
}

// Split face
{
  "action": "split",
  "source_hash": "abc123...",
  "instance_ids": [1, 2, 3],
  "new_label": "Jane Doe"
}
```

### DELETE `/api/face_library`
Delete face identity
```json
{
  "hash": "abc123..."
}
```

### GET `/serve_face/<filename>`
Serve face crop image

## Usage Workflow

### Initial Setup
1. **Process a video** with face recognition:
   - Click "🎭 Faces Only" or "📝🎭 Transcript + Faces"
   - System detects all faces automatically
   - Faces saved to database as "unlabeled"

2. **View Face Library**:
   - Go to `/face_library`
   - See all detected faces
   - Filter by: All | Labeled | Unlabeled

3. **Label Faces**:
   - Select unlabeled faces
   - Click "🏷️ Label Selected"
   - Enter person name
   - Done! Future detections will use this label

### Advanced Operations

#### Merge Similar Faces
If the system created multiple identities for the same person:
1. Select all faces of that person
2. Click "🔗 Merge Selected"
3. Choose which identity to keep
4. All instances consolidated

#### Split Incorrectly Grouped Faces
If multiple people were grouped as one identity:
1. Click on the identity
2. View all instances
3. Select instances of one person
4. Click "Split"
5. Label the new identity

## Video Processing Output

Each processed video creates `<video_name>_faces.json`:

```json
{
  "video_filename": "meeting.mp4",
  "faces_by_timestamp": {
    "12.5": [
      {"hash": "abc123...", "label": "John Doe"},
      {"hash": "def456...", "label": null}
    ],
    "45.2": [
      {"hash": "abc123...", "label": "John Doe"}
    ]
  },
  "face_hashes": ["abc123...", "def456..."],
  "labeled_people": ["John Doe"],
  "total_faces": 3
}
```

## Benefits Over Old System

### Old System (Library-First)
- Required pre-building face library from photos
- Manual photo organization
- Couldn't recognize people not in library
- All unknowns labeled as "Unknown"

### New System (Scan-First)
- ✅ No setup required - just process videos
- ✅ All faces automatically saved and tracked
- ✅ Label faces as you discover them
- ✅ Merge/split for better accuracy
- ✅ Complete history of all face detections
- ✅ Query faces by video, person, or time
- ✅ Scalable to thousands of faces

## Technical Details

### Face Matching
- Tolerance: 0.6 (configurable)
- Uses Euclidean distance on 128-d encodings
- Hash computed from rounded encoding for consistency

### Face Detection
- Two-pass system:
  1. Quick MTCNN detection every 10 seconds
  2. Detailed face_recognition on segments with faces
- Processes every 5th frame for performance
- Saves face crops at detection time

### Database Performance
- Indexed on: face_hash, video_filename, label
- Context manager for safe connections
- Atomic operations for merge/split

## Future Enhancements

Potential additions:
- Bulk labeling from external sources
- Face clustering suggestions
- Confidence scores for matching
- Export/import face library
- Face search by photo upload
- Timeline view of person appearances
- Video clips filtered by person
