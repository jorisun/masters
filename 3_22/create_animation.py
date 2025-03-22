import glob
from PIL import Image
import os
import time
import traceback

def create_animation(image_directory, output_filename='animation.gif', target_duration=30):
    """
    Create an animation from PNG files in the specified directory.
    
    Args:
        image_directory (str): Directory containing the PNG files
        output_filename (str): Name of output GIF file
        target_duration (int): Target duration of the animation in seconds
    """
    print(f"\n{'='*50}")
    print(f"Processing directory: {image_directory}")
    start_time = time.time()
    
    # Get list of PNG files sorted by frame number
    png_files = sorted(glob.glob(os.path.join(image_directory, 'final_state*.png')), 
                      key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    if not png_files:
        print(f"No PNG files found in {image_directory}")
        return
        
    num_frames = len(png_files)
    print(f"Found {num_frames} PNG files")
    
    # Calculate frame duration to achieve target animation duration
    frame_duration = int((target_duration * 1000) / num_frames)  # Convert to milliseconds
    print(f"Frame duration set to {frame_duration}ms for {target_duration}s total duration")
    
    # Open first image to get size
    print("Loading frames...")
    frames = []
    try:
        first_frame = Image.open(png_files[0])
        # Convert first frame to RGB mode and resize if needed
        first_frame = first_frame.convert('RGB')
        if first_frame.size[0] > 800:  # If width is too large
            ratio = 800 / first_frame.size[0]
            new_size = (800, int(first_frame.size[1] * ratio))
            first_frame = first_frame.resize(new_size, Image.Resampling.LANCZOS)
        
        # Append all frames
        for i, png_file in enumerate(png_files):
            frame = Image.open(png_file)
            # Convert each frame to RGB mode and resize if needed
            frame = frame.convert('RGB')
            if frame.size != first_frame.size:
                frame = frame.resize(first_frame.size, Image.Resampling.LANCZOS)
            frames.append(frame)
            if (i + 1) % 100 == 0:  # Progress update every 100 frames
                elapsed = time.time() - start_time
                print(f"Processed {i + 1}/{num_frames} frames ({((i + 1)/num_frames*100):.1f}%) - Time elapsed: {elapsed:.1f}s")
        
        print("Saving animation...")
        # Save animation with optimized settings
        first_frame.save(
            output_filename,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=True,
            quality=85
        )
        total_time = time.time() - start_time
        print(f"Animation saved as {output_filename}")
        print(f"Total processing time: {total_time:.1f}s")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error processing {image_directory}:")
        print(traceback.format_exc())
        print(f"{'='*50}")

print("\nStarting animation creation process...")
start_time = time.time()

# Create animations for each Tf directory with proper numeric sorting
directories = []
for i in range(10, 201, 10):  # Create list of expected directories
    dir_name = f'evolution_Tf{i}'
    if os.path.exists(dir_name):
        directories.append(dir_name)

print(f"Found {len(directories)} directories to process")
print(f"Expected number of frames per directory:")
for tf_dir in directories:
    if os.path.isdir(tf_dir):
        tf_value = int(tf_dir.split('Tf')[1])
        expected_frames = tf_value * 10
        print(f"  {tf_dir}: {expected_frames} frames")

# Ensure evolution_gif directory exists
gif_dir = 'evolution_gif'
if not os.path.exists(gif_dir):
    os.makedirs(gif_dir)
    print(f"\nCreated directory: {gif_dir}")

print("\nStarting processing...")

for tf_dir in directories:
    if os.path.isdir(tf_dir):
        tf_value = tf_dir.split('Tf')[1]
        output_file = os.path.join(gif_dir, f'animation_Tf{tf_value}.gif')
        create_animation(tf_dir, output_file)

total_time = time.time() - start_time
print(f"\nAnimation creation complete!")
print(f"Total execution time: {total_time:.1f}s")
print(f"All GIF files have been saved to the '{gif_dir}' directory")
