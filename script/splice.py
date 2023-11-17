import cv2
import os

# Function to split a video into frames and save them as images
def split_video_to_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0  # Initialize a counter for the frames

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If we have reached the end of the video, break the loop
        if not ret:
            break

        # Save the frame as an image with a sequential filename
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        print(f"Saved frame {frame_count} as {frame_filename}")  # Print when an image is saved

        frame_count += 1  # Increment the frame count

    # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()

    # Print a message to indicate the process is complete
    print(f"Split {frame_count} frames and saved them in {output_folder}")

if __name__ == "__main__":
    # Input video file path
    video_path = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\videos\KL.mov"
    #video_path = r"C:\Users\Sumfl\Downloads\traffic.mp4"

    # Output folder where frames will be saved as images
    output_folder = r"C:\Users\Sumfl\Downloads\extra_set_labels\set3"

    # Call the function to split the video into frames and save them
    split_video_to_frames(video_path, output_folder)
