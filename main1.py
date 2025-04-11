import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_video_frames(video_path, max_frames=20, frame_step=5):
    """
    Load frames from a video file
    
    Parameters:
    video_path (str): Path to the video file
    max_frames (int): Maximum number of frames to extract
    frame_step (int): Step size between frames (to avoid too many closely spaced frames)
    
    Returns:
    list: List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} total frames")
    
    # Calculate which frames to extract
    frame_indices = list(range(0, total_frames, frame_step))
    if len(frame_indices) > max_frames:
        # Evenly space the frames if we have too many
        frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int).tolist()
    
    print(f"Will extract {len(frame_indices)} frames")
    
    frame_count = 0
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count in frame_indices:
            # Convert from BGR to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            print(f"Loaded frame {frame_count}")
            
        frame_count += 1
        
        # Break if we've processed all frames
        if frame_count >= total_frames:
            break
            
    cap.release()
    print(f"Successfully loaded {len(frames)} frames from video")
    return frames


def denoise_frames(frames, h=10, h_color=10, template_size=7, search_size=21):
    """
    Apply denoising to reduce compression artifacts in frames
    
    Parameters:
    frames (list): List of frames to denoise
    h (int): Filter strength for luminance component (higher values remove more noise but also more details)
    h_color (int): Filter strength for color components
    template_size (int): Size of template patch used for filtering
    search_size (int): Size of search window
    
    Returns:
    list: List of denoised frames
    """
    if not frames:
        raise ValueError("No frames provided.")
        
    print("Applying denoising to remove compression artifacts...")
    
    denoised_frames = []
    for i, frame in enumerate(frames):
        # OpenCV's Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            frame, 
            None, 
            h, h_color, template_size, search_size
        )
        denoised_frames.append(denoised)
        
        if (i + 1) % 5 == 0 or i == len(frames) - 1:
            print(f"Denoised {i + 1}/{len(frames)} frames")
    
    print("Denoising complete")
    return denoised_frames


def extract_background(frames):
    """
    Extract background from frames using the median method
    
    Parameters:
    frames (list): List of frames to extract background from
    
    Returns:
    numpy.ndarray: Extracted background image
    """
    if not frames:
        raise ValueError("No frames provided.")
    
    print("Extracting background using median method...")
    
    # Create a stack of frames and compute the median
    stack = np.stack(frames, axis=0)
    background = np.median(stack, axis=0).astype(np.uint8)
    
    print("Background extraction complete")
    return background


def detect_motion(frames, background, threshold=30, kernel_size=5):
    """
    Detect motion in frames using frame differencing
    
    Parameters:
    frames (list): List of frames to detect motion in
    background (numpy.ndarray): Background image
    threshold (int): Threshold for motion detection
    kernel_size (int): Size of kernel for morphological operations
    
    Returns:
    list: List of motion masks
    """
    if not frames:
        raise ValueError("No frames provided.")
    
    print("Detecting motion using frame differencing...")
    motion_masks = []
    
    # Create kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    for i, frame in enumerate(frames):
        # Convert frame and background to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        bg_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(frame_gray, bg_gray)
        
        # Apply threshold to get binary mask
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up the mask with morphological operations
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        motion_masks.append(mask)
        
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
    
    print("Motion detection complete")
    return motion_masks


def extract_objects(frames, motion_masks):
    """
    Extract moving objects using the motion masks
    
    Parameters:
    frames (list): List of frames
    motion_masks (list): List of motion masks
    
    Returns:
    list: List of extracted object images
    """
    if not frames or not motion_masks:
        raise ValueError("Frames and motion masks must be provided.")
    
    print("Extracting moving objects...")
    extracted_objects = []
    
    for i, (frame, mask) in enumerate(zip(frames, motion_masks)):
        # Expand mask to 3 channels for RGB
        mask_3ch = np.stack([mask, mask, mask], axis=2) / 255.0
        
        # Apply mask to frame
        extracted = (frame * mask_3ch).astype(np.uint8)
        
        extracted_objects.append(extracted)
        
        if (i + 1) % 5 == 0:
            print(f"Extracted objects from {i + 1}/{len(frames)} frames")
    
    print("Object extraction complete")
    return extracted_objects


def apply_edge_smoothing(extracted_objects):
    """
    Smooth the edges of extracted objects to reduce artifacts
    
    Parameters:
    extracted_objects (list): List of extracted object images
    
    Returns:
    list: List of edge-smoothed object images
    """
    if not extracted_objects:
        raise ValueError("No extracted objects provided.")
        
    print("Smoothing object edges...")
    smoothed_objects = extracted_objects.copy()
    
    for i, obj in enumerate(extracted_objects):
        # Convert to grayscale to find edges
        gray = cv2.cvtColor(obj, cv2.COLOR_RGB2GRAY)
        
        # Find non-zero pixels (the object)
        mask = gray > 0
        
        # Create a slightly dilated mask to find the edges
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        
        # Find the edge pixels (dilated - original)
        edge_mask = dilated_mask - mask.astype(np.uint8)
        
        # Apply a blur only to the edge pixels
        blurred = cv2.GaussianBlur(obj, (5, 5), 0)
        
        # Combine the original object with blurred edges
        result = obj.copy()
        edge_indices = np.where(np.stack([edge_mask, edge_mask, edge_mask], axis=2) > 0)
        result[edge_indices] = blurred[edge_indices]
        
        smoothed_objects[i] = result
        
        if (i + 1) % 5 == 0 or i == len(extracted_objects) - 1:
            print(f"Smoothed edges for {i + 1}/{len(extracted_objects)} objects")
    
    print("Edge smoothing complete")
    return smoothed_objects


def create_composite(background, extracted_objects, frames=None, frame_indices=None, alpha=0.8):
    """
    Create the final action shot composite
    
    Parameters:
    background (numpy.ndarray): Background image
    extracted_objects (list): List of extracted object images
    frames (list): Original frames (for reference)
    frame_indices (list): Indices of frames to include in the composite. If None, use all frames.
    alpha (float): Transparency factor for blending (0-1)
    
    Returns:
    numpy.ndarray: Final composite image
    """
    if not extracted_objects:
        raise ValueError("Extracted objects must be provided.")
    
    print("Creating composite action shot...")
    
    # Start with the background
    final_composite = background.copy()
    
    # If no specific frames are specified, use all frames
    if frame_indices is None:
        frame_indices = list(range(len(extracted_objects)))
    
    # Overlay each extracted object onto the composite
    for idx in frame_indices:
        if 0 <= idx < len(extracted_objects):
            obj = extracted_objects[idx]
            
            # Create mask where object pixels are non-zero
            obj_gray = cv2.cvtColor(obj, cv2.COLOR_RGB2GRAY)
            _, obj_mask = cv2.threshold(obj_gray, 1, 255, cv2.THRESH_BINARY)
            obj_mask = obj_mask.astype(bool)
            
            # Blend the object with the background
            final_composite[obj_mask] = (
                alpha * obj[obj_mask] + (1-alpha) * final_composite[obj_mask]
            ).astype(np.uint8)
            
            print(f"Added frame {idx} to composite")
    
    # Final denoising on the composite
    final_composite = cv2.fastNlMeansDenoisingColored(
        final_composite, None, 5, 5, 7, 21
    )
    
    print("Composite creation complete")
    return final_composite


def save_composite(final_composite, output_dir="output", base_filename="action_shot"):
    """
    Save only the composite image to disk
    
    Parameters:
    final_composite (numpy.ndarray): Final composite image
    output_dir (str): Directory to save the results
    base_filename (str): Base filename for saved files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if final_composite is not None:
        # Save at high quality (95% compression quality)
        composite_path = os.path.join(output_dir, f"{base_filename}_composite.jpg")
        cv2.imwrite(composite_path, cv2.cvtColor(final_composite, cv2.COLOR_RGB2BGR), 
                   [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved final composite to {composite_path}")
        
        # Also save as PNG for lossless quality
        png_path = os.path.join(output_dir, f"{base_filename}_composite.png")
        cv2.imwrite(png_path, cv2.cvtColor(final_composite, cv2.COLOR_RGB2BGR))
        print(f"Saved lossless composite to {png_path}")
    
    return composite_path


def display_results(final_composite):
    """Display only the final composite
    
    Parameters:
    final_composite (numpy.ndarray): Final composite image
    """
    if final_composite is None:
        raise ValueError("Make sure composite is created first")
    
    # Show final result in full size
    plt.figure(figsize=(16, 9))
    plt.imshow(final_composite)
    plt.title("Final Action Shot Composite")
    plt.axis('off')
    plt.show()


def create_action_shot(video_path, output_dir="output"):
    """
    Convert a video into an action shot with denoising to reduce compression artifacts
    
    Parameters:
    video_path (str): Path to the input video
    output_dir (str): Directory to save the results
    
    Returns:
    numpy.ndarray: Final composite image
    """
    # Load frames from the video
    frames = load_video_frames(video_path, max_frames=10, frame_step=20)
    
    # Apply denoising to reduce compression artifacts
    frames = denoise_frames(frames, h=10, h_color=10, template_size=7, search_size=21)
    
    # Extract the background (median method works best for static cameras)
    background = extract_background(frames)
    
    # Detect motion using frame differencing
    motion_masks = detect_motion(frames, background, threshold=25, kernel_size=5)
    
    # Extract the moving objects
    extracted_objects = extract_objects(frames, motion_masks)
    
    # Smooth the edges of extracted objects
    smoothed_objects = apply_edge_smoothing(extracted_objects)
    
    # Create the composite action shot
    final_composite = create_composite(background, smoothed_objects, frames=frames, alpha=0.8)
    
    # Save only the composite
    save_composite(final_composite, output_dir=output_dir)
    
    # Display the results
    display_results(final_composite)
    
    return final_composite


def main():
    """Main function to process a video into an action shot"""
    # Get video path from user
    video_path = input("Enter the path to your video file: ")
    
    # Check if file exists
    if not os.path.isfile(video_path):
        print(f"Error: File {video_path} does not exist.")
        return
    
    # Get output directory
    output_dir = input("Enter output directory (default: 'output'): ") or "output"
    
    # Get base filename (without extension)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = input(f"Enter output filename (default: '{base_name}'): ") or base_name
    
    try:
        # Process the video
        print(f"Processing video: {video_path}")
        final_composite = create_action_shot(video_path, output_dir)
        
        # Save only the composite with custom filename
        save_path = save_composite(final_composite, output_dir, output_filename)
        print(f"Processing complete! Composite saved to: {save_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
