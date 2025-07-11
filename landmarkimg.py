#!/usr/bin/env python
import glob
import json
import math
import os
import sys
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from loguru import logger
from pflog import pflog
from chris_plugin import chris_plugin, PathMapper
import numpy as np

matplotlib.rcParams['font.family'] = 'monospace'
LOG = logger.debug

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> │ "
    "<level>{level: <5}</level> │ "
    "<yellow>{name: >28}</yellow>::"
    "<cyan>{function: <30}</cyan> @"
    "<cyan>{line: <4}</cyan> ║ "
    "<level>{message}</level>"
)
logger.remove()
logger.opt(colors=True)
logger.add(sys.stderr, format=logger_format)

__version__ = '1.0.5'

DISPLAY_TITLE = r"""
       _        _                 _                      _    _                 
      | |      | |               | |                    | |  (_)                
 _ __ | |______| | __ _ _ __   __| |_ __ ___   __ _ _ __| | ___ _ __ ___   __ _ 
| '_ \| |______| |/ _` | '_ \ / _` | '_ ` _ \ / _` | '__| |/ / | '_ ` _ \ / _` |
| |_) | |      | | (_| | | | | (_| | | | | | | (_| | |  |   <| | | | | | | (_| |
| .__/|_|      |_|\__,_|_| |_|\__,_|_| |_| |_|\__,_|_|  |_|\_\_|_| |_| |_|\__, |
| |                                                                        __/ |
|_|                                                                       |___/ 
"""

parser = ArgumentParser(
    description='A ChRIS plugin for marking anatomical landmarks and alignment lines on Leg X-ray images',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--inputJsonName', '-j',
                    dest='inputJsonName',
                    type=str,
                    help='Input JSON file name',
                    default='prediction.json')

parser.add_argument('--inputImageName', '-i',
                    dest='inputImageName',
                    type=str,
                    help='Name of the input image file',
                    default='leg.png')

parser.add_argument('--pointMarker', '-p',
                    dest='pointMarker',
                    type=str,
                    help='Point marker',
                    default='x')

parser.add_argument('--pointColor', '-c',
                    dest='pointColor',
                    type=str,
                    help='Color of point marker',
                    default='red')

parser.add_argument('--lineColor', '-l',
                    dest='lineColor',
                    type=str,
                    help='Color of the line to be drawn',
                    default='red')

parser.add_argument('--lineWidth', '-w',
                    dest='lineWidth',
                    type=int,
                    help='Width of lines on image',
                    default=1)

parser.add_argument('--pointSize', '-z',
                    dest='pointSize',
                    type=int,
                    help='The size of points to be plotted on the image',
                    default=10)

parser.add_argument('--outputImageExtension',
                    dest='outputImageExtension',
                    default='jpg',
                    type=str,
                    help='Generated output image file extension,'
                         'default value is jpg')

parser.add_argument('--addTextPos', '-q',
                    dest='addTextPos',
                    type=str,
                    help='Position of text placement on an input image; top or bottom',
                    default="top")

parser.add_argument('--addText',
                    dest='addText',
                    default='',
                    type=str,
                    help='optional text to add on the final image')

parser.add_argument('--addTextSize',
                    dest='addTextSize',
                    default=5.0,
                    type=float,
                    help='Size of additional text on the final output,'
                         'default value is 5.0')

parser.add_argument('--addTextColor',
                    dest='addTextColor',
                    default='white',
                    type=str,
                    help='Color of additional text on the final output,'
                         'default value is white')

parser.add_argument('-V', '--version', action='version',
                    version=f'%(prog)s {__version__}')


def preamble_show(options) -> None:
    """
    Just show some preamble "noise" in the output terminal
    """
    LOG('Version: %s' % __version__)

    LOG("plugin arguments...")
    for k, v in options.__dict__.items():
        LOG("%25s:  [%s]" % (k, v))
    LOG("")

    LOG("base environment...")
    for k, v in os.environ.items():
        LOG("%25s:  [%s]" % (k, v))
    LOG("")


# The main function of this *ChRIS* plugin is denoted by this ``@chris_plugin`` "decorator."
# Some metadata about the plugin is specified here. There is more metadata specified in setup.py.
#
# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title='A ChRIS plugin for marking anatomical landmarks',
    category='',                 # ref. https://chrisstore.co/plugins
    min_memory_limit='1000Mi',    # supported units: Mi, Gi
    min_cpu_limit='4000m',       # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0              # set min_gpu_limit=1 to enable GPU
)
def main(options: Namespace, inputdir: Path, outputdir: Path):
    """
    Generalized ChRIS plugin main method.

    - Loads landmark data from a JSON file
    - Finds and reads input images
    - Draws landmarks and connecting lines
    - Scales annotations
    - Saves annotated images and output JSONs
    """

    print(DISPLAY_TITLE)
    preamble_show(options)

    # Load input JSON data
    data = load_json_data(str(inputdir), options.inputJsonName)

    analysis_data = {}
    row_key = ""

    # Process each data entry
    for row_key, row_info in data.items():
        # 1. Locate image
        image_path = find_file_by_pattern(str(inputdir), row_key, options.inputImageName)
        image = cv2.imread(image_path)
        max_y, max_x, RGB = image.shape

        # 2. Extract landmarks
        keypoints = extract_keypoints(row_info["landmarks"])

        # 3. Set up figure and draw image
        fig = setup_figure(image)

        # 4. Scale annotation settings
        scale_annotations(fig, options)

        # 5. Draw landmarks
        draw_points(keypoints, draw_point, options)

        linePairs = [
            ("leftFemurHead", "leftAnkle"),
            ("rightFemurHead", "rightAnkle")
        ]

        # 6. Draw connecting lines
        draw_named_lines(linePairs, keypoints, draw_line, options)

        # 7. Write additional text on image
        add_positioned_text(options, max_x, max_y)

        # 8. Save annotated figure to temporary image
        temp_img_path = f"/tmp/{row_key}_img.jpg"
        save_figure_as_image(fig, temp_img_path)

        # 9. Resize and rotate image
        final_img = resize_and_rotate_image(temp_img_path, target_width=image.shape[1])

        # 10. Save final image to output directory
        output_img_path = os.path.join(outputdir, f"{row_key}.{options.outputImageExtension}")
        save_image(final_img, output_img_path)

        LOG(f"Input image dimensions: {image.shape}")
        LOG(f"Output image dimensions: {final_img.size}")

        # 11. Collect analysis data (currently empty)
        analysis_data[row_key] = {}  # Fill with real metrics or outputs if needed

    # 12. Save analysis data to JSON
    output_json_path = os.path.join(outputdir, f"{row_key}-analysis.json")
    save_json(analysis_data, output_json_path)

if __name__ == '__main__':
    main()

def draw_point(point: list, marker: str, color: str, size: float) -> None:
    """
    Draw a single point on a matplotlib figure.

    Args:
        point (list): A list or tuple of two values [x, y] representing the point's coordinates.
        marker (str): Matplotlib marker style (e.g. 'o', 'x', '.', etc.).
        color (str): Color of the marker (e.g. 'red', '#00FF00').
        size (float): Size of the marker in points².
    """
    plt.scatter(point[0], point[1], marker=marker, color=color, s=size)


def draw_line(start: list, end: list, color: str, linewidth: float) -> None:
    """
    Draw a straight line between two points on a matplotlib figure.

    Args:
        start (list): [x, y] coordinates of the line's starting point.
        end (list): [x, y] coordinates of the line's ending point.
        color (str): Line color (e.g. 'blue', '#FFAA00').
        linewidth (float): Thickness of the line in points.
    """
    x_coords = [start[0], end[0]]
    y_coords = [start[1], end[1]]
    plt.plot(x_coords, y_coords, color=color, linewidth=linewidth)

def load_json_data(inputdir: str, filename: str) -> dict:
    """
    Load and parse a JSON file from a directory (recursively).

    Args:
        inputdir (str): Base directory to search.
        filename (str): Name of the JSON file to locate.

    Returns:
        dict: Parsed JSON data.

    Raises:
        FileNotFoundError: If the JSON file is not found.
    """
    json_path = glob.glob(f"{inputdir}/**/{filename}", recursive=True)
    if not json_path:
        raise FileNotFoundError(f"{filename} not found in {inputdir}")
    LOG(f"Reading JSON file from {json_path[0]}")
    with open(json_path[0], 'r') as f:
        return json.load(f)


def find_file_by_pattern(inputdir: str, subfolder: str, filename: str) -> str:
    """
    Find a file within a nested directory structure using a pattern.

    Args:
        inputdir (str): Base directory.
        subfolder (str): Target subdirectory or identifier.
        filename (str): Filename to find.

    Returns:
        str: Full path to the found file.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    pattern = f"{inputdir}/**/{subfolder}/**/{filename}"
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {subfolder}")
    LOG(f"Found file: {matches[0]}")
    return matches[0]


def extract_keypoints(keypoint_data: list) -> dict:
    """
    Extract keypoints from structured JSON list.

    Args:
        keypoint_data (list): List of dictionaries with point data.

    Returns:
        dict: Mapping of point names to [x, y] coordinates.
    """
    keypoints = {}
    for item in keypoint_data:
        for name, coords in item.items():
            keypoints[name] = [coords["x"], coords["y"]]
    return keypoints


def setup_figure(image: np.ndarray) -> plt.Figure:
    """
    Set up a matplotlib figure and display an image.

    Args:
        image (np.ndarray): Image array (e.g., from OpenCV).

    Returns:
        plt.Figure: Configured matplotlib figure.
    """
    plt.style.use('dark_background')
    plt.axis('off')
    height, width, _ = image.shape
    fig = plt.figure(figsize=(width / 100, height / 100))
    plt.imshow(image)
    return fig


def scale_annotations(fig: plt.Figure, options) -> None:
    """
    Scale annotation parameters (e.g., text, line width) based on image size.

    Args:
        fig (plt.Figure): Matplotlib figure object.
        options: Object with annotation settings to scale (e.g., textSize, lineGap).
    """
    scale = fig.get_size_inches()[0]
    options.pointSize *= scale
    options.addTextSize *= scale


def draw_points(points: dict, draw_fn, options) -> None:
    """
    Draw keypoints on a figure using a provided draw function.

    Args:
        points (dict): Mapping of point names to [x, y] coordinates.
        draw_fn (Callable): Function to draw a single point.
        options: Drawing options (color, size, marker, etc.).
    """
    for _, point in points.items():
        draw_fn(point, options.pointMarker, options.pointColor, options.pointSize)

def add_positioned_text(options, max_x, max_y):
    """
    Adds a text annotation to a matplotlib plot based on the specified position.

    Parameters:
    ----------
    options : object
        An object with the following required attributes:
            - addTextPos (str): Position of the text. Accepts "top" or "bottom".
            - addText (str): The text string to display.
            - addTextColor (str): Color of the text.
            - addTextSize (int or float): Font size of the text.
    max_x : int or float
        The maximum x-coordinate value for the plot area.
    max_y : int or float
        The maximum y-coordinate value for the plot area.

    Raises:
    ------
    ValueError:
        If `options.addTextPos` is not "top" or "bottom".
    """
    if options.addTextPos == "top":
        x_pos = 50
        y_pos = max_y - 50
    elif options.addTextPos == "bottom":
        x_pos = max_x - 200
        y_pos = max_y - 50
    else:
        raise ValueError("Invalid addTextPos. Expected 'top' or 'bottom'.")

    plt.text(
        x_pos, y_pos, options.addText,
        color=options.addTextColor,
        fontsize=options.addTextSize,
        rotation=90
    )


def draw_named_lines(pairs: list, points: dict, draw_fn, options) -> None:
    """
    Draw lines between named point pairs.

    Args:
        pairs (list): List of (point_name_1, point_name_2) tuples.
        points (dict): Dictionary of available points.
        draw_fn (Callable): Function to draw a line between two points.
        options: Line style options (color, width, etc.).
    """
    for pt1, pt2 in pairs:
        if pt1 in points and pt2 in points:
            draw_fn(points[pt1], points[pt2], options.lineColor, options.lineWidth)


def save_figure_as_image(fig: plt.Figure, output_path: str) -> None:
    """
    Save a matplotlib figure as an image file.

    Args:
        fig (plt.Figure): The figure to save.
        output_path (str): Target file path.
    """
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.clf()


def resize_and_rotate_image(image_path: str, target_width: int, rotate_angle: int = -90) -> Image.Image:
    """
    Resize and rotate an image to match a target width.

    Args:
        image_path (str): Path to input image.
        target_width (int): Desired output width.
        rotate_angle (int): Degrees to rotate image counter-clockwise.

    Returns:
        Image.Image: Resized and rotated image.
    """
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        aspect_ratio = target_width / original_width
        new_size = (int(original_width * aspect_ratio), int(original_height * aspect_ratio))
        return img.resize(new_size).rotate(rotate_angle, expand=True)


def save_image(image: Image.Image, output_path: str) -> None:
    """
    Save a PIL Image to a specified path.

    Args:
        image (Image.Image): Image to save.
        output_path (str): Destination path including filename.
    """
    image.save(output_path)
    LOG(f"Saved image to {output_path}")


def save_json(data: dict, output_path: str) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): Data to serialize.
        output_path (str): File path to save to.
    """
    LOG(f"Saving JSON data to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
