#!/usr/bin/env python3

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # noqa:E402

import argparse
import json

import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)  # noqa:E402

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

DEFAULT_MODEL_PATH = os.path.join("model", "model.pb")
DEFAULT_LABELMAP_PATH = os.path.join("model", "labelmap.pbtxt")


def load_inference_graph(frozen_graph_path: str) -> tf.Graph:
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.io.gfile.GFile(frozen_graph_path, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

    return detection_graph


def image_to_nparray(image: Image) -> np.ndarray:
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def build_tensor_dict():
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}

    important_keys = [
        "num_detections",
        "detection_boxes",
        "detection_scores",
        "detection_classes",
    ]
    for key in important_keys:
        tensor_name = key + ":0"
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    return tensor_dict


def convert_types(output_dict):
    output_dict["num_detections"] = int(output_dict["num_detections"][0])
    output_dict["detection_classes"] = output_dict["detection_classes"][0].astype(
        np.int64
    )
    output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
    output_dict["detection_scores"] = output_dict["detection_scores"][0]


def run_session(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            tensor_dict = build_tensor_dict()
            image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

            # All outputs are float32 numpy arrays, so convert types as appropriate
            convert_types(output_dict)

    return output_dict


def output_to_json(output_from_network, thresh=0.6):
    boxes = output_from_network["detection_boxes"]
    scores = output_from_network["detection_scores"]
    classes = output_from_network["detection_classes"]

    boxes = boxes[scores > thresh]
    classes = classes[scores > thresh]
    scores = scores[scores > thresh]

    return json.dumps(
        {
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "classes": classes.tolist(),
        },
        sort_keys=True,
        indent=4,
    )


def run_inference(
    image: Image,
    model_path: str,
    labelmap_path: str,
    threshold: float,
    output_path: str = None,
) -> dict:
    """Runs model inference on provided image."""
    # The array based representation of the image will be used later in order
    # to prepare the result image with boxes and labels on it
    image_np = image_to_nparray(image)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Load frozen graph
    graph = load_inference_graph(model_path)

    # Actual detection
    output_dict = run_session(image_np_expanded, graph)

    if output_path is not None:
        category_index = label_map_util.create_category_index_from_labelmap(
            labelmap_path, use_display_name=True
        )

        # Visualization of the results of a detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict["detection_boxes"],
            output_dict["detection_classes"],
            output_dict["detection_scores"],
            category_index,
            instance_masks=output_dict.get("detection_masks"),
            use_normalized_coordinates=True,
            line_thickness=1,
        )
        dpi = 300.0
        fig = plt.figure(frameon=False, dpi=dpi)
        fig.set_size_inches(image_np.shape[1] / dpi, image_np.shape[0] / dpi)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image_np)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi="figure")
        plt.close(fig)

    return output_to_json(output_dict, threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on input image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "-o", "--output", help="If set, output image will be saved here"
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to frozen inference graph (.pb)",
    )
    parser.add_argument(
        "-l",
        "--labelmap",
        default=DEFAULT_LABELMAP_PATH,
        help="Path to frozen inference graph (.pb)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=0.6,
        type=float,
        help="Confidence threshold, will not return detections with score below this",
    )

    args = parser.parse_args()
    result = run_inference(
        Image.open(args.image),
        threshold=args.threshold,
        model_path=args.model,
        labelmap_path=args.labelmap,
        output_path=args.output,
    )
    print(result)
