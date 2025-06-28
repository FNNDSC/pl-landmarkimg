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

__version__ = '1.0.1'

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


parser = ArgumentParser(description='A ChRIS plugin for marking anatomical landmarks and alignment lines on Leg X-ray images',
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

parser.add_argument('--textColor', '-t',
                  dest='textColor',
                  type=str,
                  help='Color of text',
                  default='white')

parser.add_argument('--textSize', '-s',
                  dest='textSize',
                  type=int,
                  help='Size of the text displayed on image',
                  default=5)

parser.add_argument('--lineWidth', '-w',
                  dest='lineWidth',
                  type=int,
                  help='Width of lines on image',
                  default=1)

parser.add_argument('--textPos', '-q',
                  dest='textPos',
                  type=str,
                  help='Position of text placement on an input image; left or right',
                  default="right")

parser.add_argument('--lineGap', '-g',
                  dest='lineGap',
                  type=int,
                  help='Space between lines in pixels',
                  default=20)

parser.add_argument('--pointSize', '-z',
                  dest='pointSize',
                  type=int,
                  help='The size of points to be plotted on the image',
                  default=10)
parser.add_argument('--pftelDB',
                  dest='pftelDB',
                  default='',
                  type=str,
                  help='optional pftel server DB path')
parser.add_argument('--addText',
                  dest='addText',
                  default='',
                  type=str,
                  help='optional text to add on the final image')
parser.add_argument('--addTextPos',
                  dest='addTextPos',
                  default='right',
                  type=str,
                  help='Position of additional text on the final output,'
                       'the available choices are top, bottom, left, right and across')
parser.add_argument('--addTextSize',
                  dest='addTextSize',
                  default=5,
                  type=int,
                  help='Size of additional text on the final output,'
                       'default value is 5')
parser.add_argument('--addTextColor',
                  dest='addTextColor',
                  default='white',
                  type=str,
                  help='Color of additional text on the final output,'
                       'default value is white')
parser.add_argument('--outputImageExtension',
                  dest='outputImageExtension',
                  default='jpg',
                  type=str,
                  help='Generated output image file extension,'
                       'default value is jpg')
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
    min_memory_limit='100Mi',    # supported units: Mi, Gi
    min_cpu_limit='1000m',       # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0              # set min_gpu_limit=1 to enable GPU
)
def main(options: Namespace, inputdir: Path, outputdir: Path):
    """
    *ChRIS* plugins usually have two positional arguments: an **input directory** containing
    input files and an **output directory** where to write output files. Command-line arguments
    are passed to this main method implicitly when ``main()`` is called below without parameters.

    :param options: non-positional arguments parsed by the parser given to @chris_plugin
    :param inputdir: directory containing (read-only) input files
    :param outputdir: directory where to write output files
    """

    print(DISPLAY_TITLE)

    preamble_show(options)

    # Read json file first
    str_glob = '%s/**/%s' % (options.inputdir, options.inputJsonName)

    l_datapath = glob.glob(str_glob, recursive=True)

    jsonFilePath = l_datapath[0]

    LOG(f"Reading JSON file from {jsonFilePath}")

    f = open(jsonFilePath, 'r')
    data = json.load(f)

    d_landmarks = {}
    d_lines = {}
    d_lengths = {}
    d_json = {}
    row = ""
    for row in data:

        file_path = []
        d_info = {}
        for root, dirs, files in os.walk(options.inputdir):
            for dir in dirs:
                if dir == row:
                    dir_path = os.path.join(root, dir)
                    file_path = glob.glob(dir_path + '/**/' + options.inputImageName, recursive=True)

        LOG(f"Reading input image from {file_path[0]}")
        image = cv2.imread(file_path[0])
        # image = Image.open(file_path[0])

        plt.style.use('dark_background')
        plt.axis('off')

        max_y, max_x, RGB = image.shape
        # max_x, max_y = image.size
        fig = plt.figure(figsize=(max_x / 100, max_y / 100))
        plt.imshow(image)

        # autoscale text sizes w.r.t. image
        options.textSize = fig.get_size_inches()[0] * options.textSize
        options.addTextSize = fig.get_size_inches()[0] * options.addTextSize
        options.lineGap = fig.get_size_inches()[0] * options.lineGap
        options.pointSize = fig.get_size_inches()[0] * options.pointSize
        height = data[row]["origHeight"]
        ht_scale = height / max_x

        info = data[row]['info']
        details = data[row]['details']

        items = data[row]["landmarks"]
        for item in items:
            for i in item:
                point = [item[i]["x"], item[i]["y"]]
                d_landmarks[i] = point
                # Plot points
                draw_point(point, options.pointMarker, options.pointColor, options.pointSize)

        # Draw lines
        draw_line(d_landmarks['leftFemurHead'], d_landmarks['leftAnkle'], options.lineColor, options.lineWidth)
        draw_line(d_landmarks['rightFemurHead'], d_landmarks['rightAnkle'], options.lineColor, options.lineWidth)

        # Clean up all matplotlib stuff and save as PNG
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        plt.savefig(os.path.join("/tmp", row + "img.jpg"), bbox_inches='tight', pad_inches=0.0)
        plt.clf()

        # Open an existing image
        tmpimg = Image.open(os.path.join("/tmp", row + "img.jpg"))
        x, y = tmpimg.size
        # Calculate the aspect ratio
        aspect_ratio = max_x / x

        # Define the target width
        target_width = int(x * aspect_ratio)
        target_height = int(y * aspect_ratio)

        # Resize the image
        resized_image = tmpimg.resize((target_width, target_height))

        # Rotate the image by 90 degrees
        rotated_image = resized_image.rotate(-90, expand=True)

        # Save the resized image
        rotated_image.save(os.path.join(options.outputdir, row + f".{options.outputImageExtension}"))
        LOG(f"Input image dimensions {image.shape}")
        LOG(f"Output image dimensions {rotated_image.size}")

    jsonFilePath = os.path.join(options.outputdir, f'{row}-analysis.json')
    # Open a json writer, and use the json.dumps()
    # function to dump data
    LOG("Saving %s" % jsonFilePath)
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(d_json, indent=4))

if __name__ == '__main__':
    main()

def draw_point(point, marker, color, size):
    plt.scatter(point[0], point[1], marker=marker, color=color, s=size)

def draw_line(start, end, color, linewidth):
    X = []
    Y = []
    X.append(start[0])
    X.append(end[0])
    Y.append(start[1])
    Y.append(end[1])
    # draw connecting lines
    plt.plot(X, Y, color=color, linewidth=linewidth)

def read_image():
    pass

def save_image():
    pass
