import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional, Union, Any


def load_video_frames(video_path: str, max_frames: int = 30, frame_skip: int = 5, 
                      resize_factor: float = 1.0) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load frames from a video file
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        frame_skip: Process every nth frame
        resize_factor: Factor to resize frames (1.0 = original size)
        
    Returns:
        Tuple of (list of frames, list of original frame indices)
    """
    if not os.path.isfile(video_path):
        raise ValueError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames at {fps:.2f} FPS")
    
    # Calculate frame indices to extract
    if total_frames > max_frames * frame_skip:
        # Use evenly spaced frames
        indices = np.linspace(0, total_frames-1, max_frames, dtype=int).tolist()
    else:
        # Use every nth frame
        indices = list(range(0, total_frames, frame_skip))
        if len(indices) > max_frames:
            # If still too many, sample evenly
            indices = indices[:max_frames]
    
    frame_count = 0
    frames = []
    frame_indices = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count in indices:
            # Resize if needed
            if resize_factor != 1.0:
                height, width = frame.shape[:2]
                new_height = int(height * resize_factor)
                new_width = int(width * resize_factor)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to RGB (consistent internal format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_indices.append(frame_count)
            print(f"Loaded frame {frame_count}")
            
        frame_count += 1
        
        if frame_count >= total_frames:
            break
            
    cap.release()
    print(f"Successfully loaded {len(frames)} frames from video")
    
    return frames, frame_indices


def extract_background_mog2(frames: List[np.ndarray]) -> np.ndarray:
    """
    Extract background using MOG2 method
    
    Args:
        frames: List of video frames
        
    Returns:
        Background image
    """
    if not frames:
        raise ValueError("No frames provided for background extraction")
        
    print("Extracting background using MOG2 method...")
    
    # Create background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=len(frames),  # Use all frames for history
        varThreshold=16,      # Default threshold for detecting foreground
        detectShadows=False   # Don't detect shadows
    )
    
    # Train the model with all frames
    for i, frame in enumerate(frames):
        bg_subtractor.apply(frame)
        if (i + 1) % 10 == 0:
            print(f"Background model training: {i+1}/{len(frames)} frames")
    
    # Extract background model
    background = bg_subtractor.getBackgroundImage()
    
    print("Background extraction complete")
    return background


def detect_motion_mog2(frames: List[np.ndarray], 
                      kernel_size: int = 5) -> List[np.ndarray]:
    """
    Detect motion using MOG2 background subtraction
    
    Args:
        frames: List of video frames
        kernel_size: Size of kernel for morphological operations
        
    Returns:
        List of motion masks
    """
    if not frames:
        raise ValueError("No frames provided for motion detection")
        
    print("Detecting motion using MOG2 method...")
    
    # Create MOG2 background subtractor with short history to detect motion
    motion_detector = cv2.createBackgroundSubtractorMOG2(
        history=min(50, len(frames)),  # Shorter history for motion
        varThreshold=30,               # Default threshold
        detectShadows=False            # Don't detect shadows
    )
    
    motion_masks = []
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    for i, frame in enumerate(frames):
        # Apply background subtraction to get foreground mask
        fg_mask = motion_detector.apply(frame)
        
        # Noise reduction with morphological operations
        mask = cv2.erode(fg_mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        motion_masks.append(mask)
        
        if (i + 1) % 5 == 0:
            print(f"Motion detection: Processed {i+1}/{len(frames)} frames")
    
    print("Motion detection complete")
    return motion_masks


def extract_objects(frames: List[np.ndarray], motion_masks: List[np.ndarray], 
                   frame_indices: List[int], min_area: int = 500) -> List[Dict]:
    """
    Extract moving objects from frames using motion masks
    
    Args:
        frames: List of video frames
        motion_masks: List of motion masks
        frame_indices: Original frame indices
        min_area: Minimum area to consider as moving object
        
    Returns:
        List of dictionaries containing object data
    """
    if not frames or not motion_masks or len(frames) != len(motion_masks):
        raise ValueError("Frames and motion masks don't match")
        
    print("Extracting moving objects...")
    extracted_objects = []
    
    for i, (frame, mask) in enumerate(zip(frames, motion_masks)):
        # Skip frames with little motion
        if cv2.countNonZero(mask) < min_area:
            continue
            
        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask.copy(), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter small contours
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if significant_contours:
            # Create a refined mask with only significant contours
            refined_mask = np.zeros_like(mask)
            cv2.drawContours(refined_mask, significant_contours, -1, 255, -1)
            
            # Extract the object
            mask_3ch = np.stack([refined_mask, refined_mask, refined_mask], axis=2) / 255.0
            extracted = (frame * mask_3ch).astype(np.uint8)
            
            obj_data = {
                'frame': frame,
                'mask': refined_mask,
                'extracted': extracted,
                'index': i,
                'orig_index': frame_indices[i] if i < len(frame_indices) else i
            }
            
            extracted_objects.append(obj_data)
            
        if (i + 1) % 5 == 0:
            print(f"Object extraction: Processed {i+1}/{len(frames)} frames")
            
    print(f"Extracted {len(extracted_objects)} moving objects")
    return extracted_objects


def smooth_edges(objects: List[Dict]) -> List[Dict]:
    """
    Apply edge smoothing to extracted objects for better blending
    
    Args:
        objects: List of object dictionaries from extract_objects
        
    Returns:
        List of objects with smoothed edges
    """
    if not objects:
        raise ValueError("No objects to smooth")
        
    print("Smoothing object edges...")
    smoothed_objects = []
    
    for i, obj in enumerate(objects):
        frame = obj['frame']
        mask = obj['mask']
        
        # Create a dilated version of the mask to find edge regions
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Edge mask is the difference between dilated and original
        edge_mask = dilated_mask - mask
        
        # Apply Gaussian blur to the frame
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Create smoothed object by using original in mask area and blurred in edge area
        result = frame.copy()
        
        # Create 3-channel edge mask
        edge_mask_3ch = np.stack([edge_mask, edge_mask, edge_mask], axis=2) > 0
        
        # Replace edge pixels with blurred version
        result[edge_mask_3ch] = blurred[edge_mask_3ch]
        
        # Update the object data
        smoothed_obj = obj.copy()
        smoothed_obj['smoothed'] = result
        smoothed_objects.append(smoothed_obj)
        
        if (i + 1) % 5 == 0:
            print(f"Edge smoothing: Processed {i+1}/{len(objects)} objects")
            
    print("Edge smoothing complete")
    return smoothed_objects


def create_composite(background: np.ndarray, objects: List[Dict], alpha: float = 0.8) -> np.ndarray:
    """
    Create final composite image by overlaying objects onto background
    
    Args:
        background: Background image
        objects: List of object dictionaries
        alpha: Alpha blending factor (0.0-1.0)
        
    Returns:
        Final composite image
    """
    if background is None or not objects:
        raise ValueError("Need background and objects for composition")
        
    print(f"Creating composite action shot with {len(objects)} objects...")
    
    # Start with a copy of the background
    canvas = background.copy()
    
    # Sort objects by frame index for consistent layering
    objects.sort(key=lambda x: x['orig_index'])
    
    # Composite objects onto background
    for obj in objects:
        # Get mask and object (use smoothed if available)
        mask = obj['mask']
        frame_obj = obj.get('smoothed', obj['extracted'])
        
        # Create alpha mask for blending
        mask_float = mask.astype(float) / 255.0
        alpha_mask = np.stack([mask_float, mask_float, mask_float], axis=2)
        
        # Alpha blend: result = alpha * obj + (1-alpha) * background
        canvas_region = alpha * frame_obj + (1-alpha) * canvas
        
        # Apply only where mask is non-zero
        canvas = np.where(alpha_mask > 0, canvas_region, canvas).astype(np.uint8)
        
    print("Composite creation complete")
    return canvas


def save_result(final_composite: np.ndarray, output_dir: str = "output", 
               filename: str = "action_shot") -> Tuple[str, str]:
    """
    Save the final composite image
    
    Args:
        final_composite: Final composite image
        output_dir: Directory to save the output
        filename: Base filename for the output (without extension)
        
    Returns:
        Tuple of paths to the saved JPG and PNG files
    """
    if final_composite is None:
        raise ValueError("No composite available to save")
        
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save as JPEG (smaller file)
    jpg_path = os.path.join(output_dir, f"{filename}.jpg")
    bgr_image = cv2.cvtColor(final_composite, cv2.COLOR_RGB2BGR)
    cv2.imwrite(jpg_path, bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved JPEG version to {jpg_path}")
    
    # Save as PNG (lossless)
    png_path = os.path.join(output_dir, f"{filename}.png")
    cv2.imwrite(png_path, bgr_image)
    print(f"Saved lossless PNG version to {png_path}")
    
    return jpg_path, png_path


def display_result(final_composite: np.ndarray) -> None:
    """
    Display the final action shot composite
    
    Args:
        final_composite: Final composite image
    """
    if final_composite is None:
        raise ValueError("No composite available to display")
        
    plt.figure(figsize=(16, 9))
    plt.imshow(final_composite)
    plt.title("Final Action Shot Composite")
    plt.axis('off')
    plt.show()


def create_action_shot(video_path: str, output_dir: str = "output", 
                       max_frames: int = 30, frame_skip: int = 5,
                       resize_factor: float = 1.0,
                       alpha_blend: float = 0.8,
                       min_object_area: int = 500) -> np.ndarray:
    """
    Create an action shot from a video using MOG2 for background and motion detection
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save output files
        max_frames: Maximum number of frames to process
        frame_skip: Process every nth frame
        resize_factor: Factor to resize frames (1.0 = original size)
        alpha_blend: Alpha value for blending objects (0.0-1.0)
        min_object_area: Minimum area to consider as moving object
        
    Returns:
        Final composite image
    """
    try:
        # Step 1: Load frames from video
        frames, frame_indices = load_video_frames(
            video_path, 
            max_frames=max_frames, 
            frame_skip=frame_skip,
            resize_factor=resize_factor
        )
        
        # Step 2: Extract background using MOG2
        background = extract_background_mog2(frames)
        
        # Step 3: Detect motion using MOG2
        motion_masks = detect_motion_mog2(frames, kernel_size=5)
        
        # Step 4: Extract moving objects
        extracted_objects = extract_objects(
            frames, 
            motion_masks, 
            frame_indices,
            min_area=min_object_area
        )
        
        # Step 5: Smooth object edges
        smoothed_objects = smooth_edges(extracted_objects)
        
        # Step 6: Create final composite
        final_composite = create_composite(
            background, 
            smoothed_objects, 
            alpha=alpha_blend
        )
        
        # Step 7: Save the result
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_filename}_action"
        save_result(final_composite, output_dir, output_filename)
        
        return final_composite
        
    except Exception as e:
        print(f"Error creating action shot: {str(e)}")
        raise


def main():
    """Process a video to create an action shot"""
    # Example video path
    video_path = "video.mp4"
    
    if not os.path.isfile(video_path):
        print(f"Error: File {video_path} does not exist.")
        return
    
    # Output directory
    output_dir = "output"
    
    try:
        print(f"Processing video: {video_path}")
        
        # Create action shot with MOG2 methods
        final_composite = create_action_shot(
            video_path=video_path,
            output_dir=output_dir,
            max_frames=20,             # Maximum frames to process
            frame_skip=20,             # Process every 10th frame
            resize_factor=0.75,        # Resize to 75% for faster processing
            alpha_blend=0.8,           # Alpha blending factor
            min_object_area=500        # Minimum area for objects
        )
        
        # Display the result
        display_result(final_composite)
        
        print("Processing complete!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()