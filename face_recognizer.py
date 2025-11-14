#!/usr/bin/env python3
"""
Face Recognition Module for Video Library
Implements two-pass face detection and recognition system
"""

import os
import json
import pickle
import time
import cv2
import numpy as np
from pathlib import Path
import face_recognition
from mtcnn import MTCNN
import threading
from typing import Dict, List, Tuple, Optional


class FaceLibrary:
    """Manages the face library with known faces and their encodings"""
    
    def __init__(self, library_dir: str = None):
        if library_dir is None:
            # Default to face_library folder in video directory
            from video_viewer_app import get_video_directory
            video_dir = get_video_directory()
            library_dir = os.path.join(video_dir, "face_library")
        
        self.library_dir = library_dir
        self.library_file = os.path.join(library_dir, "face_library.pkl")
        self.photos_dir = os.path.join(library_dir, "photos")
        self.known_faces = {}  # {name: [encodings]}
        
        # Ensure directories exist
        os.makedirs(library_dir, exist_ok=True)
        os.makedirs(self.photos_dir, exist_ok=True)
        
        self.load_library()
    
    def add_person(self, name: str, face_encoding: np.ndarray) -> bool:
        """Add a face encoding for a person"""
        try:
            if name not in self.known_faces:
                self.known_faces[name] = []
            self.known_faces[name].append(face_encoding.tolist())  # Convert to list for JSON serialization
            self.save_library()
            return True
        except Exception as e:
            print(f"Error adding person {name}: {e}")
            return False
    
    def remove_person(self, name: str) -> bool:
        """Remove a person from the library"""
        try:
            if name in self.known_faces:
                del self.known_faces[name]
                self.save_library()
                return True
            return False
        except Exception as e:
            print(f"Error removing person {name}: {e}")
            return False
    
    def rename_person(self, old_name: str, new_name: str) -> bool:
        """Rename a person in the library"""
        try:
            if old_name in self.known_faces and new_name not in self.known_faces:
                self.known_faces[new_name] = self.known_faces[old_name]
                del self.known_faces[old_name]
                self.save_library()
                return True
            return False
        except Exception as e:
            print(f"Error renaming person {old_name} to {new_name}: {e}")
            return False
    
    def find_matches(self, unknown_encoding: np.ndarray, tolerance: float = 0.6) -> List[str]:
        """Find matching persons for an unknown face encoding"""
        matches = []
        try:
            for name, encodings in self.known_faces.items():
                # Convert back to numpy arrays for comparison
                known_encodings = [np.array(enc) for enc in encodings]
                results = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance)
                if any(results):
                    matches.append(name)
        except Exception as e:
            print(f"Error finding matches: {e}")
        
        return matches
    
    def get_all_persons(self) -> List[str]:
        """Get list of all known persons"""
        return list(self.known_faces.keys())
    
    def get_person_count(self, name: str) -> int:
        """Get number of face encodings for a person"""
        return len(self.known_faces.get(name, []))
    
    def save_library(self):
        """Save face library to file"""
        try:
            with open(self.library_file, 'wb') as f:
                pickle.dump(self.known_faces, f)
        except Exception as e:
            print(f"Error saving face library: {e}")
    
    def load_library(self):
        """Load face library from file"""
        try:
            if os.path.exists(self.library_file):
                with open(self.library_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
            else:
                self.known_faces = {}
        except Exception as e:
            print(f"Error loading face library: {e}")
            self.known_faces = {}
    
    def build_from_photos(self, progress_callback=None):
        """Build face library from organized photo folders"""
        results = {"added": 0, "errors": []}
        
        if not os.path.exists(self.photos_dir):
            return results
        
        person_dirs = [d for d in os.listdir(self.photos_dir) 
                      if os.path.isdir(os.path.join(self.photos_dir, d))]
        
        total_dirs = len(person_dirs)
        
        for i, person_name in enumerate(person_dirs):
            if progress_callback:
                progress_callback(f"Processing {person_name}...", int((i / total_dirs) * 100))
            
            person_dir = os.path.join(self.photos_dir, person_name)
            photo_files = [f for f in os.listdir(person_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for photo_file in photo_files:
                try:
                    photo_path = os.path.join(person_dir, photo_file)
                    image = face_recognition.load_image_file(photo_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    for encoding in encodings:
                        self.add_person(person_name, encoding)
                        results["added"] += 1
                        
                except Exception as e:
                    results["errors"].append(f"Error processing {photo_file} for {person_name}: {str(e)}")
        
        if progress_callback:
            progress_callback("Completed!", 100)
        
        return results


class FaceRecognizer:
    """Main face recognition processor for videos"""
    
    def __init__(self, face_library: FaceLibrary = None):
        self.face_library = face_library or FaceLibrary()
        self.detector = MTCNN()
        
    def has_faces_in_segment(self, video_path: str, start_time: float) -> bool:
        """Quick face detection check for a video segment (Pass 1)"""
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return False
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(rgb_frame)
            
            return len(faces) > 0
            
        except Exception as e:
            print(f"Error checking faces in segment at {start_time}s: {e}")
            return False
    
    def extract_faces_from_segment(self, video_path: str, start_time: float, duration: float = 10) -> List[Dict]:
        """Detailed face recognition for a video segment (Pass 2)"""
        faces_found = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            
            end_time = start_time + duration
            frame_count = 0
            
            while cap.get(cv2.CAP_PROP_POS_MSEC) < end_time * 1000:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 5th frame to balance accuracy and performance
                frame_count += 1
                if frame_count % 5 != 0:
                    continue
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                
                for encoding, location in zip(face_encodings, face_locations):
                    matches = self.face_library.find_matches(encoding)
                    
                    if not matches:
                        matches = ["Unknown"]
                    
                    faces_found.append({
                        'timestamp': current_time,
                        'location': location,
                        'matches': matches,
                        'encoding': encoding.tolist()  # Store for potential future use
                    })
            
            cap.release()
            
        except Exception as e:
            print(f"Error extracting faces from segment {start_time}-{start_time + duration}: {e}")
        
        return faces_found
    
    def process_video(self, video_path: str, task_id: str = None, progress_callback=None) -> Dict:
        """Process entire video with two-pass face recognition"""
        results = {
            'video_path': video_path,
            'faces_by_timestamp': {},
            'people_found': set(),
            'total_faces': 0,
            'processing_time': 0
        }
        
        start_processing = time.time()
        
        try:
            # Get video duration
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            if progress_callback:
                progress_callback(f"Starting face detection (duration: {duration:.1f}s)", 5)
            
            # Pass 1: Quick face detection every 10 seconds
            face_segments = []
            segment_interval = 10
            total_segments = int(duration / segment_interval) + 1
            
            for i, start_time in enumerate(range(0, int(duration), segment_interval)):
                if progress_callback:
                    progress = 5 + int((i / total_segments) * 30)  # 5-35%
                    progress_callback(f"Pass 1: Checking segment {i+1}/{total_segments}", progress)
                
                if self.has_faces_in_segment(video_path, start_time):
                    face_segments.append(start_time)
            
            if progress_callback:
                progress_callback(f"Pass 1 complete. Found {len(face_segments)} segments with faces", 35)
            
            # Pass 2: Detailed analysis of face-containing segments
            for i, start_time in enumerate(face_segments):
                if progress_callback:
                    progress = 35 + int((i / len(face_segments)) * 60)  # 35-95%
                    progress_callback(f"Pass 2: Analyzing segment {i+1}/{len(face_segments)}", progress)
                
                faces = self.extract_faces_from_segment(video_path, start_time)
                
                for face_data in faces:
                    timestamp = face_data['timestamp']
                    matches = face_data['matches']
                    
                    if timestamp not in results['faces_by_timestamp']:
                        results['faces_by_timestamp'][timestamp] = []
                    
                    results['faces_by_timestamp'][timestamp].extend(matches)
                    results['people_found'].update(matches)
                    results['total_faces'] += 1
            
            # Convert set to list for JSON serialization
            results['people_found'] = list(results['people_found'])
            results['processing_time'] = time.time() - start_processing
            
            if progress_callback:
                progress_callback(f"Complete! Found {results['total_faces']} faces, {len(results['people_found'])} people", 100)
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            results['error'] = str(e)
        
        return results


def save_face_recognition_data(video_path: str, face_data: Dict):
    """Save face recognition data to JSON file"""
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(video_path)
        json_path = os.path.join(video_dir, f"{video_name}_faces.json")
        
        # Add metadata
        face_data['created_at'] = time.time()
        face_data['video_file'] = os.path.basename(video_path)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(face_data, f, indent=2, ensure_ascii=False)
        
        return json_path
        
    except Exception as e:
        print(f"Error saving face recognition data: {e}")
        return None


def load_face_recognition_data(video_path: str) -> Optional[Dict]:
    """Load face recognition data from JSON file"""
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(video_path)
        json_path = os.path.join(video_dir, f"{video_name}_faces.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
        
    except Exception as e:
        print(f"Error loading face recognition data: {e}")
        return None


def process_video_face_recognition(video_path: str, task_id: str, processing_status: Dict):
    """Background task for face recognition processing"""
    try:
        processing_status[task_id] = {
            "status": "initializing",
            "progress": 0,
            "details": "Loading face recognition models..."
        }
        
        # Initialize face recognizer
        face_recognizer = FaceRecognizer()
        
        def update_progress(message, progress):
            processing_status[task_id] = {
                "status": "processing",
                "progress": progress,
                "details": message
            }
        
        # Process video
        results = face_recognizer.process_video(video_path, task_id, update_progress)
        
        if 'error' in results:
            processing_status[task_id] = {
                "status": "error",
                "message": results['error']
            }
            return
        
        # Save results
        json_path = save_face_recognition_data(video_path, results)
        
        processing_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "details": f"Found {results['total_faces']} faces, {len(results['people_found'])} people",
            "results": {
                "total_faces": results['total_faces'],
                "people_found": results['people_found'],
                "processing_time": results['processing_time'],
                "json_file": os.path.basename(json_path) if json_path else None
            }
        }
        
    except Exception as e:
        processing_status[task_id] = {
            "status": "error",
            "message": f"Face recognition error: {str(e)}"
        }


if __name__ == "__main__":
    # Test the face recognition system
    print("Face Recognition System Test")
    
    # Initialize face library
    face_lib = FaceLibrary()
    print(f"Face library initialized with {len(face_lib.get_all_persons())} known persons")
    
    # Initialize face recognizer
    recognizer = FaceRecognizer(face_lib)
    print("Face recognizer initialized")
