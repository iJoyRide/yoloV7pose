import cv2
import numpy as np
import argparse
import os

def read_class_names(file_path):
    class_names = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip() == '':
                    continue
                class_id, class_name = line.strip().split(': ')
                class_names[int(class_id)] = class_name
    except FileNotFoundError:
        print(f"Class names file {file_path} not found.")
    except ValueError as e:
        print(f"Error parsing class names file: {e}")
    return class_names

def visualize(img, bbox_array, class_names):
    h, w, _ = img.shape

    for bbox in bbox_array:
        class_id = int(bbox[0])
        score = bbox[5]  # Assuming the score is at index 5
        # Convert bbox coordinates from normalized to absolute values
        x_center, y_center, bw, bh = bbox[1:5]
        x_center, y_center, bw, bh = [x_center * w, y_center * h, bw * w, bh * h]
        xmin = int(x_center - bw / 2)
        ymin = int(y_center - bh / 2)
        xmax = int(x_center + bw / 2)
        ymax = int(y_center + bh / 2)

        class_name = class_names.get(class_id, "Unknown")

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (105, 237, 249), 2)
        label = f"{class_name} {score:.2f}"
        cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 2)
    
    return img


def main(input_path, output_path):
    class_names = read_class_names('labels.txt')

    for file_name in os.listdir(input_path):
        if file_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(input_path, file_name)
            annotation_path = image_path.rsplit('.', 1)[0] + '.txt'

            if not os.path.exists(annotation_path):
                print(f"No corresponding annotation found for {file_name}. Skipping file.")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image {image_path}. Skipping file.")
                continue

            with open(annotation_path, 'r') as f:
                annos = [list(map(float, line.split())) for line in f.readlines()]

            if not annos:
                print(f"No annotations found in {annotation_path}. Skipping file.")
                continue

            drawn_image = visualize(image, annos, class_names)
            output_file = os.path.join(output_path, 'visualized_' + os.path.basename(image_path))
            cv2.imwrite(output_file, drawn_image)
            print(f"Saved visualized image to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default=r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\test", 
                    required=False, help="The directory path to the images + labels.")
    parser.add_argument('--output', '-o', type=str, default=r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\test\output",
                    required=False, help="The directory path to save the visualized images.")

    
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    main(input_path, output_path)
