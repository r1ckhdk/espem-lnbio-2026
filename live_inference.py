import argparse
import time

import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors


# -------------------------
# Argumentos
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Live inference with YOLOv8.")
    parser.add_argument(
        "--webcam_index", type=int, default=0, help="Index of the webcam to use."
    )
    return parser.parse_args()


# -------------------------
# Modelos
# -------------------------
def load_models():
    model_seg = YOLO("yolov8n-seg.pt", task="segment")
    model_pose = YOLO("yolov8n-pose.pt", task="pose")
    return model_seg, model_pose


# -------------------------
# Utilidades
# -------------------------
def compute_fps(start, end):
    return int(1 / (end - start)) if end > start else 0


def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


# -------------------------
# Inferência
# -------------------------
def run_segmentation(model, image, device):
    return model.predict(
        image,
        # 0: person
        # 16: dog
        # 24: backpack
        # 25: umbrella
        # 26: handbag
        # 39: bottle
        # 41: cup
        # 47: apple
        # 56: chair
        # 62: tv
        # 67: cell phone
        # 73: book
        # 74: clock
        # To list all classes: model.names
        classes=[0, 24, 25, 26, 39, 41, 47, 56, 62, 67, 73, 74],
        device=device,
    )


def run_pose(model, image, device):
    return model.predict(image, device=device)


# -------------------------
# Renderizacao
# -------------------------
def render_segmentation_panel(results, shape, colors):
    h, w, _ = shape
    panel = np.zeros((h, w, 3), dtype=np.uint8)

    cv.putText(
        panel,
        "Segmentacao",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    res = results[0]
    if res.masks is None or res.boxes is None:
        return panel

    masks = res.masks.data.cpu().numpy()

    for i, box in enumerate(res.boxes):
        cls = int(box.cls[0])
        color = colors(cls, True)

        mask = cv.resize(
            masks[i],
            (w, h),
            interpolation=cv.INTER_NEAREST,
        )

        panel[mask > 0.5] = color

    return panel


def render_pose_panel(results_pose, shape):
    h, w, _ = shape
    panel = np.zeros((h, w, 3), dtype=np.uint8)

    if results_pose and results_pose[0].keypoints is not None:
        panel = results_pose[0].plot(
            img=panel, boxes=False, kpt_radius=5, kpt_line=True
        )

    cv.putText(
        panel,
        "Faca uma pose",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return panel


def resize_and_pad(image, target_w, target_h):
    h, w = image.shape[:2]
    src_ratio = w / h
    target_ratio = 16 / 9

    if src_ratio > target_ratio:
        new_w = target_w
        new_h = int(target_w / src_ratio)
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = pad_right = 0
    else:
        new_h = target_h
        new_w = int(target_h * src_ratio)
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = pad_bottom = 0

    image = cv.resize(image, (new_w, new_h))
    return cv.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    device = get_device()

    model_seg, model_pose = load_models()
    colors = Colors()

    cap = cv.VideoCapture(args.webcam_index)

    panel_w, panel_h = 960, 540  # cada painel (16:9)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.flip(frame, 1)
            start = time.perf_counter()

            seg_results = run_segmentation(model_seg, frame, device)
            pose_results = run_pose(model_pose, frame, device)

            end = time.perf_counter()
            fps = compute_fps(start, end)

            # Painel 1: imagem crua
            panel_raw = frame.copy()
            cv.putText(
                panel_raw,
                f"FPS: {fps}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Painel 2: segmentacao
            panel_seg = render_segmentation_panel(seg_results, frame.shape, colors)

            # Painel 3: deteccao
            panel_det = seg_results[0].plot(boxes=True)
            cv.putText(
                panel_det,
                "Deteccao",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            # Painel 4: pose
            panel_pose = render_pose_panel(pose_results, frame.shape)

            # Ajuste de tamanho
            panels = [
                resize_and_pad(p, panel_w, panel_h)
                for p in (panel_raw, panel_seg, panel_det, panel_pose)
            ]

            top = cv.hconcat(panels[:2])
            bottom = cv.hconcat(panels[2:])
            grid = cv.vconcat([top, bottom])

            cv.namedWindow("Multi-view Display", cv.WND_PROP_FULLSCREEN)
            cv.setWindowProperty(
                "Multi-view Display",
                cv.WND_PROP_FULLSCREEN,
                cv.WINDOW_FULLSCREEN,
            )
            cv.imshow("Multi-view Display", grid)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
