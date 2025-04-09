import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class VideoToActionShot:
    def __init__(self):
        self.frames = []
        self.background = None
        self.motion_masks = []
        self.extracted_objects = []
        self.final_composite = None
        
    def load_video_frames(self, video_path, max_frames=20, frame_step=5):
        """
        Load frames from a video file
        
        Parameters:
        video_path (str): Path to the video file
        max_frames (int): Maximum number of frames to extract
        frame_step (int): Step size between frames (to avoid too many closely spaced frames)
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
        self.frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count in frame_indices:
                # Convert from BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame_rgb)
                print(f"Loaded frame {frame_count}")
                
            frame_count += 1
            
            # Break if we've processed all frames
            if frame_count >= total_frames:
                break
                
        cap.release()
        print(f"Successfully loaded {len(self.frames)} frames from video")
        return self.frames
    
    def denoise_frames(self, h=10, h_color=10, template_size=7, search_size=21):
        """
        Apply denoising to reduce compression artifacts in frames
        
        Parameters:
        h (int): Filter strength for luminance component (higher values remove more noise but also more details)
        h_color (int): Filter strength for color components
        template_size (int): Size of template patch used for filtering
        search_size (int): Size of search window
        """
        if not self.frames:
            raise ValueError("No frames loaded. Call load_video_frames first.")
            
        print("Applying denoising to remove compression artifacts...")
        
        denoised_frames = []
        for i, frame in enumerate(self.frames):
            # OpenCV's Non-local Means Denoising
            denoised = cv2.fastNlMeansDenoisingColored(
                frame, 
                None, 
                h, h_color, template_size, search_size
            )
            denoised_frames.append(denoised)
            
            if (i + 1) % 5 == 0 or i == len(self.frames) - 1:
                print(f"Denoised {i + 1}/{len(self.frames)} frames")
        
        self.frames = denoised_frames
        print("Denoising complete")
        return self.frames
    
    def extract_background(self, method="median"):
        """Extract background from frames using median or average method"""
        if not self.frames:
            raise ValueError("No frames loaded. Call load_video_frames first.")
        
        print(f"Extracting background using {method} method...")
        
        if method == "median":
            # Create a stack of frames and compute the median
            stack = np.stack(self.frames, axis=0)
            self.background = np.median(stack, axis=0).astype(np.uint8)
        elif method == "average":
            # Create a stack of frames and compute the average
            stack = np.stack(self.frames, axis=0)
            self.background = np.mean(stack, axis=0).astype(np.uint8)
        elif method == "first_frame":
            # Use the first frame as background
            self.background = self.frames[0].copy()
        else:
            raise ValueError(f"Unknown background extraction method: {method}")
        
        print("Background extraction complete")
        return self.background
    
    def detect_motion(self, method="frame_diff", threshold=30, kernel_size=5):
        """Detect motion in frames using frame differencing or background subtraction"""
        if not self.frames:
            raise ValueError("No frames loaded. Call load_video_frames first.")
            
        if method == "frame_diff" and self.background is None:
            print("No background available, extracting background first...")
            self.extract_background()
        
        print(f"Detecting motion using {method} method...")
        self.motion_masks = []
        
        # Create kernel for morphological operations
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        for i, frame in enumerate(self.frames):
            if method == "frame_diff":
                # Convert frame and background to grayscale
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                bg_gray = cv2.cvtColor(self.background, cv2.COLOR_RGB2GRAY)
                
                # Calculate absolute difference
                diff = cv2.absdiff(frame_gray, bg_gray)
                
                # Apply threshold to get binary mask
                _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                
            elif method == "MOG2":
                # Create background subtractor
                bg_subtractor = cv2.createBackgroundSubtractorMOG2()
                
                # Convert to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Apply background subtraction
                mask = bg_subtractor.apply(frame_bgr)
                
            else:
                raise ValueError(f"Unknown motion detection method: {method}")
            
            # Clean up the mask with morphological operations
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            self.motion_masks.append(mask)
            
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(self.frames)} frames")
        
        print("Motion detection complete")
        return self.motion_masks
    
    def extract_objects(self):
        """Extract moving objects using the motion masks"""
        if not self.frames or not self.motion_masks:
            raise ValueError("Frames and motion masks must be available. Detect motion first.")
        
        print("Extracting moving objects...")
        self.extracted_objects = []
        
        for i, (frame, mask) in enumerate(zip(self.frames, self.motion_masks)):
            # Expand mask to 3 channels for RGB
            mask_3ch = np.stack([mask, mask, mask], axis=2) / 255.0
            
            # Apply mask to frame
            extracted = (frame * mask_3ch).astype(np.uint8)
            
            self.extracted_objects.append(extracted)
            
            if (i + 1) % 5 == 0:
                print(f"Extracted objects from {i + 1}/{len(self.frames)} frames")
        
        print("Object extraction complete")
        return self.extracted_objects
    
    def apply_edge_smoothing(self):
        """Smooth the edges of extracted objects to reduce artifacts"""
        if not self.extracted_objects:
            raise ValueError("No extracted objects available.")
            
        print("Smoothing object edges...")
        
        for i, obj in enumerate(self.extracted_objects):
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
            
            self.extracted_objects[i] = result
            
            if (i + 1) % 5 == 0 or i == len(self.extracted_objects) - 1:
                print(f"Smoothed edges for {i + 1}/{len(self.extracted_objects)} objects")
        
        print("Edge smoothing complete")
        return self.extracted_objects
    
    def create_composite(self, frame_indices=None, alpha=0.8):
        """
        Create the final action shot composite
        
        Parameters:
        frame_indices (list): Indices of frames to include in the composite. If None, use all frames.
        alpha (float): Transparency factor for blending (0-1)
        """
        if not self.frames or not self.extracted_objects:
            raise ValueError("Frames and extracted objects must be available.")
        
        if self.background is None:
            print("No background available, extracting background first...")
            self.extract_background()
        
        print("Creating composite action shot...")
        
        # Start with the background
        self.final_composite = self.background.copy()
        
        # If no specific frames are specified, use all frames
        if frame_indices is None:
            frame_indices = list(range(len(self.frames)))
        
        # Overlay each extracted object onto the composite
        for idx in frame_indices:
            if 0 <= idx < len(self.extracted_objects):
                obj = self.extracted_objects[idx]
                
                # Create mask where object pixels are non-zero
                obj_gray = cv2.cvtColor(obj, cv2.COLOR_RGB2GRAY)
                _, obj_mask = cv2.threshold(obj_gray, 1, 255, cv2.THRESH_BINARY)
                obj_mask = obj_mask.astype(bool)
                
                # Blend the object with the background
                self.final_composite[obj_mask] = (
                    alpha * obj[obj_mask] + (1-alpha) * self.final_composite[obj_mask]
                ).astype(np.uint8)
                
                print(f"Added frame {idx} to composite")
        
        # Final denoising on the composite
        self.final_composite = cv2.fastNlMeansDenoisingColored(
            self.final_composite, None, 5, 5, 7, 21
        )
        
        print("Composite creation complete")
        return self.final_composite
    
    def save_results(self, output_dir="output", base_filename="action_shot"):
        """Save the results to disk"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if self.background is not None:
            bg_path = os.path.join(output_dir, f"{base_filename}_background.jpg")
            cv2.imwrite(bg_path, cv2.cvtColor(self.background, cv2.COLOR_RGB2BGR))
            print(f"Saved background to {bg_path}")
            
        if self.final_composite is not None:
            # Save at high quality (95% compression quality)
            composite_path = os.path.join(output_dir, f"{base_filename}_composite.jpg")
            cv2.imwrite(composite_path, cv2.cvtColor(self.final_composite, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Saved final composite to {composite_path}")
            
            # Also save as PNG for lossless quality
            png_path = os.path.join(output_dir, f"{base_filename}_composite.png")
            cv2.imwrite(png_path, cv2.cvtColor(self.final_composite, cv2.COLOR_RGB2BGR))
            print(f"Saved lossless composite to {png_path}")
            
        # Save a few sample motion masks and extracted objects
        if self.motion_masks and self.extracted_objects:
            for i in range(min(3, len(self.motion_masks))):
                mask_path = os.path.join(output_dir, f"{base_filename}_mask_{i}.jpg")
                cv2.imwrite(mask_path, self.motion_masks[i])
                
                obj_path = os.path.join(output_dir, f"{base_filename}_object_{i}.jpg")
                cv2.imwrite(obj_path, cv2.cvtColor(self.extracted_objects[i], cv2.COLOR_RGB2BGR))
            
            print(f"Saved sample masks and objects to {output_dir}")
    
    def display_results(self):
        """Display the original frames, background, and final composite"""
        if not self.frames or self.final_composite is None:
            raise ValueError("Make sure frames are loaded and composite is created first")
        
        # Display background and final composite
        plt.figure(figsize=(15, 10))
        
        plt.subplot(1, 2, 1)
        plt.imshow(self.background)
        plt.title("Extracted Background")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(self.final_composite)
        plt.title("Final Action Shot Composite")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Display a few original frames and their corresponding motion masks
        num_frames = min(4, len(self.frames))
        indices = np.linspace(0, len(self.frames)-1, num_frames, dtype=int)
        
        plt.figure(figsize=(15, 8))
        for i, idx in enumerate(indices):
            # Original frame
            plt.subplot(2, num_frames, i+1)
            plt.imshow(self.frames[idx])
            plt.title(f"Frame {idx}")
            plt.axis('off')
            
            # Motion mask
            plt.subplot(2, num_frames, i+1+num_frames)
            plt.imshow(self.motion_masks[idx], cmap='gray')
            plt.title(f"Motion Mask {idx}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Show final result in full size
        plt.figure(figsize=(16, 9))
        plt.imshow(self.final_composite)
        plt.title("Final Action Shot Composite")
        plt.axis('off')
        plt.show()


# Example usage
def create_action_shot(video_path, output_dir="output"):
    """
    Convert a video into an action shot with denoising to reduce compression artifacts
    
    Parameters:
    video_path (str): Path to the input video
    output_dir (str): Directory to save the results
    """
    # Create the action shot processor
    processor = VideoToActionShot()
    
    # Load frames from the video
    processor.load_video_frames(video_path, max_frames=15, frame_step=8)
    
    # Apply denoising to reduce compression artifacts
    processor.denoise_frames(h=10, h_color=10, template_size=7, search_size=21)
    
    # Extract the background (median method works best for static cameras)
    processor.extract_background(method="median")
    
    # Detect motion using frame differencing
    processor.detect_motion(method="frame_diff", threshold=25, kernel_size=5)
    
    # Extract the moving objects
    processor.extract_objects()
    
    # Smooth the edges of extracted objects
    processor.apply_edge_smoothing()
    
    # Create the composite action shot
    processor.create_composite(alpha=0.8)
    
    # Save the results
    processor.save_results(output_dir=output_dir)
    
    # Display the results
    processor.display_results()
    
    return processor.final_composite


if __name__ == "__main__":
    # Specify your video path here
    video_path = "video.mp4"  # Replace with your video file path
    
    # Create the action shot
    action_shot = create_action_shot(video_path)
