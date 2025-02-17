#!/usr/bin/env python3
'''
This code is created by chatGPT by other code files, and not test yet
'''
import os
import cv2
import numpy as np
import argparse
import librosa
import moviepy.editor as mp

def extract_frames(video_path, output_folder, target_fps=25):
    """
    Extract frames from a video and save them as PNG images.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder where extracted frames will be saved.
        target_fps (int): Desired frame rate for extraction.
    
    Returns:
        int: Total number of saved frames.
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(round(video_fps / target_fps)), 1)
    count = 0
    saved_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"{saved_frame_count}.png")
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1
        count += 1
    cap.release()
    print(f"Extracted {saved_frame_count} frames from video.")
    return saved_frame_count

def extract_audio(video_path, output_audio_path):
    """
    Extract audio from a video and save it as a WAV file.
    """
    video_clip = mp.VideoFileClip(video_path)
    video_clip.audio.write_audiofile(output_audio_path, logger=None)
    video_clip.close()
    print(f"Audio extracted and saved to: {output_audio_path}")

def compute_mel(audio_path, sr=16000, n_mels=80, hop_length=200):
    """
    Compute a mel spectrogram from an audio file using Librosa.

    Returns:
        np.ndarray: Mel spectrogram with shape (n_mels, time)
    """
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    print(f"Computed mel spectrogram with shape: {mel_spec.shape}")
    return mel_spec

# --- Stub functions for pose and emotion extraction ---

def extract_pose_from_frame(frame):
    """
    Extract pose features from a single frame.
    
    Replace this stub with your actual pose estimation implementation
    (for example, using MediaPipe or OpenPose).
    
    Returns:
        np.ndarray: A 1D array representing pose features (e.g., 20-dim).
    """
    pose_vector = np.random.rand(20).astype(np.float32)
    return pose_vector

def extract_emotion_from_frame(frame):
    """
    Extract emotion features from a single frame.
    
    Replace this stub with your actual facial emotion recognition implementation.
    
    Returns:
        np.ndarray: A 1D array representing emotion features (e.g., 10-dim).
    """
    emotion_vector = np.random.rand(10).astype(np.float32)
    return emotion_vector

def extract_pose_and_emotion(frames_folder, num_frames):
    """
    Process all extracted frames to compute per-frame pose and emotion features.
    
    Returns:
        tuple: Two numpy arrays containing pose features and emotion features,
               each with shape (num_frames, feature_dim).
    """
    pose_list = []
    emotion_list = []
    for i in range(num_frames):
        frame_path = os.path.join(frames_folder, f"{i}.png")
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
        pose_vector = extract_pose_from_frame(frame)
        emotion_vector = extract_emotion_from_frame(frame)
        pose_list.append(pose_vector)
        emotion_list.append(emotion_vector)
    
    pose_arr = np.array(pose_list)  # shape: (num_frames, 20)
    emotion_arr = np.array(emotion_list)  # shape: (num_frames, 10)
    print(f"Extracted pose array shape: {pose_arr.shape}")
    print(f"Extracted emotion array shape: {emotion_arr.shape}")
    return pose_arr, emotion_arr

def prepare_video_dataset(video_path, output_dir, target_fps=25):
    """
    Prepare the dataset for a single video by extracting frames, audio,
    and computing mel, pose, and emotion features.

    The output directory will contain:
      - A folder 'frames/' with images (0.png, 1.png, …)
      - mel.npy (mel spectrogram, shape: [n_mels, time])
      - pose.npy (pose features, shape: [num_frames, feature_dim])
      - emotion_face.npy (emotion features, shape: [num_frames, feature_dim])
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract frames.
    frames_folder = os.path.join(output_dir, "frames")
    num_frames = extract_frames(video_path, frames_folder, target_fps=target_fps)
    
    # Step 2: Extract audio and compute mel spectrogram.
    audio_path = os.path.join(output_dir, "audio.wav")
    extract_audio(video_path, audio_path)
    mel_spec = compute_mel(audio_path)
    # Save mel.npy with the expected shape (n_mels, time). The training code will transpose it.
    np.save(os.path.join(output_dir, "mel.npy"), mel_spec)
    
    # Step 3: Extract pose and emotion features from frames.
    pose_arr, emotion_arr = extract_pose_and_emotion(frames_folder, num_frames)
    np.save(os.path.join(output_dir, "pose.npy"), pose_arr)
    np.save(os.path.join(output_dir, "emotion_face.npy"), emotion_arr)
    print(f"Dataset prepared at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset from a real video file.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed dataset for this video.")
    parser.add_argument("--target_fps", type=int, default=25, help="Frame rate for extracting frames (default: 25).")
    args = parser.parse_args()
    
    prepare_video_dataset(args.video, args.output_dir, target_fps=args.target_fps)
