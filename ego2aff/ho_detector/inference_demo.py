import cv2
import os

# Define the path to the folder containing videos
folder_path = '/home/gen/Ego4d/data/v2/clips/'

# List all files in the folder
video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mkv'))]

# Iterate over each video file
for video_file in video_files:
    video_path = os.path.join(folder_path, video_file)
    
    # Debugging: Check if the file exists
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        continue
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}.")
        continue

    print(f"Processing video: {video_file}")

    # Read frames from the video
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # # Display the resulting frame
        # cv2.imshow('Frame', frame)
        
        # # Press Q on the keyboard to exit the loop
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

    # Optional: Add a short delay to ensure proper resource release
    # cv2.waitKey(100)

# Close all the frames
cv2.destroyAllWindows()
