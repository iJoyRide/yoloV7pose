import argparse
import time
from pathlib import Path

import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    TracedModel,
)


def calculate_center(x, y, width, height):  # for 17 key points of object
    center_x = x + width / 2
    center_y = y + height / 2
    return center_x, center_y


def process_line(line):  # for 17 key points of object
    values = line.strip().split()
    x, y, width, height = map(float, values[1:])
    center_x, center_y = calculate_center(x, y, width, height)
    coord_str = f"{center_x:.6f} {center_y:.6f} 2 "
    new_line = f"{line.strip()} {coord_str * 17}\n"
    return new_line


def get_save_path(base_dir: Path, input_base_dir: Path, input_path: Path) -> Path:
    relative_path = input_path.relative_to(input_base_dir)
    # print(f"Input Base Dir: {input_base_dir}") # Debugging
    # print(f"Input Path: {input_path}")         # Debugging
    # print(f"Relative Path: {relative_path}")   # Debugging
    return base_dir / relative_path


def detect(save_img=True):
    source, weights, view_img, save_txt, imgsz, trace = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.img_size,
        not opt.no_trace,
    )
    save_img = not opt.nosave and not source.endswith(".txt")  # save inference images
    webcam = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    )

    input_base_dir = opt.source  # This will be whatever you provide in the command line
    project_dir = (
        opt.project
    )  # 'runs/detect' or any other directory where you want to save results

    # Directories
    save_dir = Path(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    )  # increment run
    print(save_dir)
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != "cpu" and (
            old_img_b != img.shape[0]
            or old_img_h != img.shape[2]
            or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            p = Path(p)  # Convert to Path

            # Determine the text save path:
            txt_save_path = get_save_path(save_dir, input_base_dir, p).with_suffix(
                ".txt"
            )
            txt_save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Save Path: {txt_save_path}")
            txt_path = str(txt_save_path)
            # Ensure the directories exist before saving text results
            s += "%gx%g " % img.shape[2:]  # print string

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if cls.item() == 0:  # change to the ID u want
                        cls = torch.tensor(
                            7, device=cls.device
                        )  # change to the ID u want
                    elif cls.item() == 1:  # change to the ID u want
                        cls = torch.tensor(
                            6, device=cls.device
                        )  # change to the ID u want
                    elif cls.item() == 2:  # change to the ID u want
                        cls = torch.tensor(
                            8, device=cls.device
                        )  # change to the ID u want
                    elif cls.item() == 3:  # change to the ID u want
                        cls = torch.tensor(
                            2, device=cls.device
                        )  # change to the ID u want
                    elif cls.item() == 5:  # change to the ID u want
                        cls = torch.tensor(
                            9, device=cls.device
                        )  # change to the ID u want
                    elif cls.item() == 7:  # change to the ID u want
                        cls = torch.tensor(
                            10, device=cls.device
                        )  # change to the ID u want
                    # elif cls.item() == 8:  # change to the ID u want
                    #     cls = torch.tensor(
                    #         12, device=cls.device
                    #     )  # change to the ID u want
                    # elif cls.item() == 9:  # change to the ID u want
                    #     cls = torch.tensor(
                    #         13, device=cls.device
                    #     )  # change to the ID u want

                    if save_txt:  # Write to file
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )  # normalized xywh
                        line = (
                            (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        )  # label format
                        formatted_line = ("%g " * len(line)).rstrip() % line
                        processed_line = process_line(formatted_line)
                        with open(txt_path, "a") as f:
                            f.write(processed_line)

                    if save_img or view_img:  # Add bbox to image
                        label = f"{names[int(cls)]} {conf:.2f}" if int(cls) < len(names) else f"Unknown {conf:.2f}"

                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors[int(cls) % len(colors)],
                            line_thickness=1,
                        )

            # Print time (inference + NMS)
            print(
                f"{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS"
            )

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #         print(f" The image with the result is saved in: {save_path}")
            #     else:  # 'video' or 'stream'
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path += '.mp4'
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer.write(im0)

    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )
        # print(f"Results saved to {save_dir}{s}")

    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument(
        "--weights", nargs="+", type=str, default="best.pt", help="model.pt path(s)"
    )
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument("--source", type=str, default="test/test3/vid4", help="source")
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.35, help="object confidence threshold" #higher more accurate
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.25, help="IOU threshold for NMS" #lower the less overlapping boundary boxes
    )
    parser.add_argument(
        "--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--no-trace", action="store_true", help="don`t trace model")
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["best.pt"]:  # changeable
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
# python detect_objects.py --weights best.pt --source D:\sumfei\yolov8pose\data\images\train --save-txt --classes 0
# python detect_objects.py --weights best.pt --source X:\Yolo\YoloV8CY\yolov8pose\data\images\train --save-txt --classes 0
# python detect_objects.py --weights best.pt --source \\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data\images\train --save-txt --classes 0 2 3 4 6 8 9 --device 0
# python detect_objects.py --weights best.pt --source /data/images/train --save-txt --classes 0 2 3 4 6 8 9 --device 0
