#!/usr/bin/env python
"""
#-USAGE------------------------------------
 | Provides some common usual utilites (drawing and processing images, logger, ...)
"""
import os
import logging

logerror = logging.error
loginfo = logging.info
import random
import math
import json
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from base64 import b64encode, b64decode
import msgpack_numpy as m
from msgpack_numpy import encode
from msgpack_numpy import decode
from msgpack import unpackb
from msgpack import packb

m.patch()

# from rich import print
# from rich import pretty
# pretty.install()
# ["Rich and pretty", True]


# def rich_logging_conf_docker():
#     from rich.console import Console
#     from rich.logging import RichHandler
FORMAT = f"%(msecs)03d-| %(funcName)-25s | - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]")
# handlers=[RichHandler(rich_tracebacks=True,
# console=Console(width=150),
# locals_max_string=150)])


# def rich_logging_conf_comp():
#     from rich.logging import RichHandler
#     FORMAT = f"%(msecs)03d-| %(funcName)-25s | - %(message)s"
#     logging.basicConfig(level=logging.INFO,
#                         format=FORMAT,
#                         datefmt="[%X]",
#                         handlers=[RichHandler(rich_tracebacks=True,
#                                             markup=True)])


# try:
#     rich_config = os.environ['RICH_CONFIG']
#     if rich_config == 'TRUE':
#         rich_logging_conf_comp()
#     else:
#         rich_logging_conf_docker()
#         #rich_logging_conf_comp()
# except:
#     rich_logging_conf_docker()
#     #rich_logging_conf_comp()


class NumpyEncoder(json.JSONEncoder):
    """
    #-USAGE------------------------------------
        | Nedded to correctly dumping numpy array to JSON.
        |
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def image_zip(image):
    """
    #-USAGE------------------------------------
        | Serialize data using msgpack.
    #-ARGUMENTS--------------------------------
        | - image    - Image (ndarray).
    """
    image = packb(image, default=m.encode)
    return image


def image_unzip(image):
    """
    #-USAGE------------------------------------
        | Deserialize data using msgpack.
    #-ARGUMENTS--------------------------------
        | - image    - Image (ndarray).
    """
    try:
        image = unpackb(image, object_hook=m.decode)
    except:
        raise RuntimeError("Could not decode the contents")
    return image


def encode_ndarray_msgpack(images):
    """
    #-USAGE------------------------------------
        | Encode list of ndarray(or list with shape as image nd.array) to BYTES msgpack.
    #-ARGUMENTS--------------------------------
        | - images    - List of images (ndarrays or lists). Format: [[N, IMAGE, IMAGE, ... ], [N, IMAGE, IMAGE, ... ], ... ]
    """
    result_encoded = []
    for images_ in images:
        numbr = images_[0]
        encoded_images = [numbr]
        for image in images_[1:]:
            image = np.uint8(image)
            encoded_images.append(image_zip(image))
        result_encoded.append(encoded_images)
    """ RETURN: [[N, BYTES, BYTES, ... ], ... ] """
    return result_encoded


# pythran export decode_msgpack_ndarray(list list)
def decode_msgpack_ndarray(data):
    """
    #-USAGE------------------------------------
        | Decode list of BYTES from msgpack to ndarray.uint8.
    #-ARGUMENTS--------------------------------
        | - data    - List of base64.utf-8. Format: [[N, BYTES, BYTES, ... ], ... ]
    """
    result_decoded = []
    for data_ in data:
        numbr = data_[0]
        decoded_images = [numbr]
        for image in data_[1:]:
            image = image_unzip(image)
            decoded_images.append(image)
        result_decoded.append(decoded_images)
    """ RETURN: [[N, np.uint8_IMAGE, np.uint8_IMAGE, ... ], ... ] """
    return result_decoded


# --------------------------------------------
# numpy(single) [0, 1] <--->  numpy(unit)
# --------------------------------------------
def uint2single(img):

    return np.float32(img / 255.0)


def single2uint(img):

    return np.uint8((img.clip(0, 1) * 255.0).round())


def uint162single(img):

    return np.float32(img / 65535.0)


def single2uint16(img):

    return np.uint16((img.clip(0, 1) * 65535.0).round())


def convert_to_boxes(result):
    n_boxes = len(result["level"])
    boxes = []
    for i in range(n_boxes):
        (x, y, w, h) = (
            result["left"][i],
            result["top"][i],
            result["width"][i],
            result["height"][i],
        )
        box = [[x, y], [(x + w), y], [(x + w), (y + h)], [x, (y + h)]]
        boxes.append(box)
    return boxes


def draw_box_polygon(draw_obj, box, color):
    """
    #-USAGE------------------------------------
        | Drawing polygon with cordinates from BOX after text detecting.
    """
    draw_obj.polygon(
        (
            box[0][0],
            box[0][1],
            box[1][0],
            box[1][1],
            box[2][0],
            box[2][1],
            box[3][0],
            box[3][1],
        ),
        outline=color,
    )


def draw_box_polyline(draw_obj, box, width=2, color="blue"):
    """
    #-USAGE------------------------------------
        | Drawing polyline from cordinates from BOX after text detecting.
        |
    #-ARGUMENTS--------------------------------
        | - draw_obj    ImageDraw.Draw
        | - box         [[0, 1], [2, 3], [4, 5], [6, 7]]
        | - width       Width of lines
        | - color       (R, G, B) or key string 'red', 'blue' ...
    """
    # Needed 5 point:
    points = (
        (box[0][0], box[0][1]),
        (box[1][0], box[1][1]),
        (box[2][0], box[2][1]),
        (box[3][0], box[3][1]),
        (box[0][0], box[0][1]),
    )

    draw_obj.line(points, fill=color, width=width)
    for point in points:
        draw_obj.ellipse(
            (
                point[0] - int(width / 2),
                point[1] - int(width / 2),
                point[0] + int(width / 2),
                point[1] + int(width / 2),
            ),
            fill=color,
        )


def draw_ocr_box(image, boxes, scores=None):
    """
    #-MAIN-DATA--------------------------------
        | input:    np.array    (H, W, C) | RGB
        | output:   np.array    (H, W, C) | RGB
    #-USAGE------------------------------------
        | Drawing boxes around detecting text on image.
    #-ARGUMENTS--------------------------------
        | - image       np.array (H, W, C)
        | - boxes       [[[0, 1], [2, 3], [4, 5], [6, 7]], ...]
        | - scores      [INT, INT, INT, ...]
    """
    # Need convert to PIL Image to can drawing
    image = Image.fromarray(image).convert("RGB")

    img_left = image.copy()

    loginfo(f" | BEGIN ->")
    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    for idx, box in enumerate(boxes):
        if scores == None:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        else:
            color = (int((100 - scores[idx]) * 2.5), 0, int(scores[idx] * 2.5))
        draw_box_polyline(draw_left, box, color=color)
    img_left = Image.blend(image, img_left, 0.5)

    loginfo(f" |-> DONE")
    return np.array(img_left)


def draw_ocr_box_txt(
    image,
    boxes,
    texts,
    scores=None,
    drop_score=50,
    font_path="/app/PaddleOCR/doc/fonts/simfang.ttf",
):
    """
    #-MAIN-DATA--------------------------------
        | input:    np.array    (H, W, C) | RGB
        | output:   np.array    (H, W, C) | RGB
    #-USAGE------------------------------------
        | Drawing boxes around detecting text on image.
        | Drawing boxes and text into boxes on white BG image.
        | Condcatenate images and return as np.array
    #-ARGUMENTS--------------------------------
        | - image       np.array (H, W, C)
        | - boxes       [[[0, 1], [2, 3], [4, 5], [6, 7]], ...]
        | - texts       ['RECOGNIZED_TEXT', ...]
        | - scores      [0.1124, ...]
        | - drop_score  0=< <=100 - dont include boxes with text score < drop_score
        | - font_path   "/" - set path to font for drawings text
    """
    loginfo(f" | BEGIN ->")
    # Original image (draw only boxes)
    img_left = Image.fromarray(draw_ocr_box(image, boxes)).convert("RGB")

    # Need convert to PYL Image to can drawing
    image = Image.fromarray(image).convert("RGB")
    h, w = image.height, image.width

    # Addition image with white BG  (draw boxes with text)
    img_right = Image.new("RGB", (w, h), (255, 255, 255))

    # Lock random on certain values
    random.seed(0)

    # Drawing boxes + text on right image
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, texts)):
        if (scores is not None) and (scores[idx] < drop_score):
            continue
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # color = (150, 150, 150)
        draw_box_polygon(draw_right, box, color)
    for idx, (box, txt) in enumerate(zip(boxes, texts)):
        if (scores is not None) and (scores[idx] < drop_score):
            continue
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        box_height = math.sqrt(
            (box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2
        )
        box_width = math.sqrt(
            (box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2
        )
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.7), 10)
            font = ImageFont.truetype(
                font_path,
                font_size,
                encoding="utf-8",
                layout_engine=ImageFont.LAYOUT_BASIC,
            )
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text((box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.6), 10)
            font = ImageFont.truetype(
                font_path,
                font_size,
                encoding="utf-8",
                layout_engine=ImageFont.LAYOUT_BASIC,
            )
            draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)

    # Create new image with ((255, 255, 255) BG + BOXES + TEXTS) on right and initial image on left:
    img_show = Image.new("RGB", (int(w * 2.3), h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))

    loginfo(f" |-> DONE")
    return np.array(img_show)


def draw_ocr_txt(
    image,
    boxes,
    texts,
    scores=None,
    drop_score=50,
    font_path="/app/PaddleOCR/doc/fonts/simfang.ttf",
):
    """
    #-MAIN-DATA--------------------------------
        | input:    np.array    (H, W, C) | RGB
        | output:   np.array    (H, W, C) | RGB
    #-USAGE------------------------------------
        | Drawing boxes around detecting text on image.
        | Drawing boxes and text into boxes on white BG image.
        | Condcatenate images and return as np.array
    #-ARGUMENTS--------------------------------
        | - image       np.array (H, W, C)
        | - boxes       [[[0, 1], [2, 3], [4, 5], [6, 7]], ...]
        | - texts       ['RECOGNIZED_TEXT', ...]
        | - scores      [0.1124, ...]
        | - drop_score  0=< <=100 - dont include boxes with text score < drop_score
        | - font_path   "/" - set path to font for drawings text
    """
    loginfo(f" | BEGIN ->")
    # Original image (draw only boxes)
    img_left = Image.fromarray(draw_ocr_box(image, boxes, scores)).convert("RGB")

    # Need convert to PYL Image to can drawing
    image = Image.fromarray(image).convert("RGB")
    h, w = image.height, image.width

    # Addition image with white BG  (draw boxes with text)
    img_right = Image.new("RGB", (w, h), (255, 255, 255))

    # Lock random on certain values
    random.seed(0)

    def pretty_str(d, indent=2):
        str_dd = ""
        for key, value in d.items():
            str_dd += "\n" + "  " * indent + str(key) + "\n"
            if isinstance(value, dict):
                pretty_str(value, indent + 1)
            else:
                str_dd += "  " * (indent + 1) + str(value) + "\n"
        return str_dd

    texts = [pretty_str(texts)]
    boxes = [[[0, 0], [int(w / 35), 0], [int(w / 35), int(h / 35)], [0, int(h / 35)]]]

    # Drawing boxes + text on right image
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, texts)):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # color = (150, 150, 150)
        draw_box_polygon(draw_right, box, color)
    for idx, (box, txt) in enumerate(zip(boxes, texts)):
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        box_height = math.sqrt(
            (box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2
        )
        box_width = math.sqrt(
            (box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2
        )
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.7), 10)
            font = ImageFont.truetype(
                font_path,
                font_size,
                encoding="utf-8",
                layout_engine=ImageFont.LAYOUT_BASIC,
            )
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text((box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.6), 10)
            font = ImageFont.truetype(
                font_path,
                font_size,
                encoding="utf-8",
                layout_engine=ImageFont.LAYOUT_BASIC,
            )
            draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)

    # Create new image with ((255, 255, 255) BG + BOXES + TEXTS) on right and initial image on left:
    img_show = Image.new("RGB", (int(w * 2.3), h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))

    loginfo(f" |-> DONE")
    return np.array(img_show)


def image_boxing(image, result, font_path=""):
    """
    #-MAIN-DATA--------------------------------
        | input:    np.array    (H, W, C) | RGB
        | output:   np.array    (H, W, C) | RGB
    #-USAGE------------------------------------
        | Drawing all information on image from result after OCR recognition.
    #-ARGUMENTS--------------------------------
        | - image       np.array (H, W, C) | RGB
        | - result      output from PaddleOCR or TesseractOCR
        | - font_path   "/" -  path to font
    #-INTO-RESULT------------------------------
        | boxes       [[[0, 1], [2, 3], [4, 5], [6, 7]], ...]
        | texts       ['RECOGNIZED_TEXT', ...]
        | scores      [0.1124, ...]
    """
    loginfo(f" | BEGIN (IMAGE{image.shape}, BOXES LEN:{len(result)}) -> ")

    boxes = []
    texts = []
    scors = []

    # Check if we have bad values and bad boxes after OCR:
    boxes, texts, scors = check_data_image(result)

    # Drawing detecting BOXES with castom font if set.
    if font_path != "":
        image = draw_ocr_box_txt(image, boxes, texts, font_path=font_path)
    else:
        image = draw_ocr_box_txt(image, boxes, texts)

    loginfo(f" |-> DONE")
    # np.array    (H, W, C) | RGB
    return image


def image_textng(image, result, text_res, font_path=""):
    """
    #-MAIN-DATA--------------------------------
        | input:    np.array    (H, W, C) | RGB
        | output:   np.array    (H, W, C) | RGB
    #-USAGE------------------------------------
        | Drawing all information on image from result after OCR recognition.
    #-ARGUMENTS--------------------------------
        | - image       np.array (H, W, C) | RGB
        | - result      output from PaddleOCR or TesseractOCR
        | - font_path   "/" -  path to font
    #-INTO-RESULT------------------------------
        | boxes       [[[0, 1], [2, 3], [4, 5], [6, 7]], ...]
        | texts       ['RECOGNIZED_TEXT', ...]
        | scores      [0.1124, ...]
    """
    loginfo(f" | BEGIN (IMAGE{image.shape}, BOXES LEN:{len(result)}) -> ")

    boxes = []
    texts = []
    scors = []

    # Check if we have bad values and bad boxes after OCR:
    boxes, texts, scors = check_data_image(result, image_shape=image.shape)

    # Drawing detecting BOXES with castom font if set.
    if font_path != "":
        image = draw_ocr_txt(image, boxes, text_res, scors, font_path=font_path)
    else:
        image = draw_ocr_txt(image, boxes, text_res, scors)

    loginfo(f" |-> DONE")
    # np.array    (H, W, C) | RGB
    return image


def read_image(file_path):
    """
    #-USAGE------------------------------------
        | Read image from path.
        | return np.uint8 (H, W, C) | RGB
    """
    fc = file_path[-5:]
    # Read PDF file as PIL Image:
    if ".pdf" in fc:
        from pdf2image import convert_from_path
        import psutil

        num_cpus = psutil.cpu_count()  # logical=False - for only real cpu cores
        image = convert_from_path(
            file_path, fmt="jpeg", dpi=300, thread_count=num_cpus
        )[0]
        image = np.uint8(image)
    elif (
        ".bmp" in fc
        or ".dib" in fc
        or ".jpeg" in fc
        or ".jpg" in fc
        or ".jpe" in fc
        or ".jp2" in fc
        or ".png" in fc
        or ".pbm" in fc
        or ".pgm" in fc
        or ".ppm" in fc
        or ".pxm" in fc
        or ".pnm" in fc
        or ".pfm" in fc
        or ".sr" in fc
        or ".ras" in fc
        or ".tiff" in fc
        or ".tif" in fc
        or ".exr" in fc
        or ".hdr" in fc
        or ".pic" in fc
        or ".webp" in fc
    ):
        # Read image (H, W, C) BGR
        image = cv2.imread(file_path)
        # image = np.uint8(Image.open(file_path).quantize(colors=8, method=2).convert('RGB')) # Dont speed up

        if image.ndim == 2:  # dont work...
            image = np.expand_dims(image, axis=2)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.uint8(image)
    logging.info(f" | READ IMAGE... \t{image.shape}")
    # Check image and convert to necessary format (H, W, C) RGB
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    # Dont speed up
    # H_TARGET = 1300
    # H_DIFF = 200
    # (h, w, c) = image.shape
    # if h > H_TARGET:
    #     coeff_res = h / (H_TARGET - H_DIFF)
    #     # Resize image. FORMAT dsize (W, H) !!!!
    #     image = cv2.resize(image, dsize=(int(w/coeff_res), int(h/coeff_res)), interpolation=cv2.INTER_CUBIC)

    # np.array (H, W, C) | RGB
    return np.uint8(image)


def morphology(img):
    # inverts the image to execute easier operations (sum and subtraction)
    a, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    # generates 3 by 3 cross kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # iteration counter
    iteration = 0
    while 1:
        iteration += 1
        # erosion
        last_img = img.copy()
        ero = cv2.erode(img, kernel, iterations=1)
        # dilation
        dil = cv2.dilate(ero, kernel, iterations=1)
        # result = original - dilated + eroded
        img -= dil
        img += ero
        # ends loop if result is the same from last iteration
        if cv2.compare(img, last_img, cv2.CMP_EQ).all():
            break
    a, img = cv2.threshold(
        img, 100, 255, cv2.THRESH_BINARY_INV
    )  # inverts back the image
    return img


def eraseTwoByTwos(img):
    if img.ndim == 3:
        altura, largura, canal = img.shape
    elif img.ndim == 2:
        altura, largura = img.shape
    obj = 0
    bg = 255
    for y in range(1, altura - 2):
        for x in range(1, largura - 2):
            # centrais
            c1 = img[y, x]
            c2 = img[y, x + 1]
            c3 = img[y + 1, x]
            c4 = img[y + 1, x + 1]
            if (c1 == obj) and (c2 == obj) and (c3 == obj) and (c4 == obj):
                if img[y - 1, x - 1] != obj:
                    img[y, x] = bg
                    pass
                elif img[y - 1, x + 2] != obj:
                    img[y, x + 1] = bg
                    pass
                elif img[y + 2, x - 1] != obj:
                    img[y + 1, x] = bg
                    pass
                elif img[y + 2, x + 2] != obj:
                    img[y + 1, x + 1] = bg
                    pass
                # vizinhos
                v1 = img[y - 1, x - 1]
                v2 = img[y - 1, x]
                v3 = img[y - 1, x + 1]
                v4 = img[y - 1, x + 2]
                v5 = img[y, x + 2]
                v6 = img[y + 1, x + 2]
                v7 = img[y + 2, x + 2]
                v8 = img[y + 2, x + 1]
                v9 = img[y + 2, x]
                v10 = img[y + 2, x - 1]
                v11 = img[y + 1, x - 1]
                v12 = img[y, x - 1]
                vizinhos = [v12, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11]


def cleanCorners(img):
    altura = img.shape[0]
    largura = img.shape[1]
    bg = 255
    img[0:altura, 0] = bg
    img[0, 0:largura] = bg
    img[0:altura, largura - 1] = bg
    img[altura - 1, 0:largura] = bg


def eraseLadders(img):
    altura = img.shape[0]
    largura = img.shape[1]
    obj = 0
    bg = 255
    m1 = [[255, 0, 7], [0, 0, 7], [7, 7, 255]]
    m2 = [[7, 0, 255], [7, 0, 0], [255, 7, 7]]
    m3 = [[7, 7, 255], [0, 0, 7], [255, 0, 7]]
    m4 = [[255, 7, 7], [7, 0, 0], [7, 0, 255]]
    mask = [m1, m2, m3, m4]
    for y in range(1, altura - 1):
        for x in range(1, largura - 1):
            p5 = img[y, x]
            if p5 == obj:
                p1 = img[y - 1, x - 1]
                p2 = img[y - 1, x]
                p3 = img[y - 1, x + 1]
                p4 = img[y, x - 1]
                p6 = img[y, x + 1]
                p7 = img[y + 1, x - 1]
                p8 = img[y + 1, x]
                p9 = img[y + 1, x + 1]
                p = [[p1, p2, p3], [p4, p5, p6], [p7, p8, p9]]
                for m in mask:
                    pairing = 1
                    for i in range(0, 3):
                        for j in range(0, 3):
                            if m[i][j] != 7:
                                if m[i][j] != p[i][j]:
                                    pairing = 0
                    if pairing:
                        img[y, x] = bg
                        break


def adaptative_thresholding(image, threshold):

    I = image

    # Convert image to grayscale
    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    # Original image size
    orignrows, origncols = gray.shape

    # Windows size
    M = int(np.floor(orignrows / 16) + 1)
    N = int(np.floor(origncols / 16) + 1)

    # Image border padding related to windows size
    Mextend = round(M / 2) - 1
    Nextend = round(N / 2) - 1

    # Padding image
    aux = cv2.copyMakeBorder(
        gray,
        top=Mextend,
        bottom=Mextend,
        left=Nextend,
        right=Nextend,
        borderType=cv2.BORDER_REFLECT,
    )

    windows = np.zeros((M, N), np.int32)

    # Image integral calculation
    imageIntegral = cv2.integral(aux, windows, -1)

    # Integral image size
    nrows, ncols = imageIntegral.shape

    # Memory allocation for cumulative region image
    result = np.zeros((orignrows, origncols))

    # Image cumulative pixels in windows size calculation
    for i in range(nrows - M):
        for j in range(ncols - N):

            result[i, j] = (
                imageIntegral[i + M, j + N]
                - imageIntegral[i, j + N]
                + imageIntegral[i, j]
                - imageIntegral[i + M, j]
            )

    # Output binary image memory allocation
    binar = np.ones((orignrows, origncols), dtype=np.bool)

    # Gray image weighted by windows size
    graymult = (gray).astype("float64") * M * N

    # Output image binarization
    binar[graymult <= result * (100.0 - threshold) / 100.0] = False

    # binary image to UINT8 conversion
    binar = (255 * binar).astype(np.uint8)

    return binar


def nice_proc(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Determine average contour area
    average_area = []
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        average_area.append(area)
    average = sum(average_area) / len(average_area)
    # Remove large lines if contour area is 5x bigger then average contour area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > average * 5:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
    # Dilate with vertical kernel to connect characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=3)
    # Remove small noise if contour area is smaller than 4x average
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < average * 4:
            cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)
    # Bitwise mask with input image
    result = cv2.bitwise_and(image, image, mask=dilate)
    result[dilate == 0] = (255, 255, 255)
    image = result

    return image


def sauvola_threshold(img, w_size=15, k=0.35):
    """Sauvola's thresholding algorithm.
    - w_size    INT                     The size of the local window to compute each pixel threshold. Should be and odd value
    - k         interval [0.2, 0.5]     Controls the value of the local threshold."""
    # Obtaining rows and cols
    rows, cols = img.shape
    i_rows, i_cols = rows + 1, cols + 1

    # Computing integral images
    # Leaving first row and column in zero for convenience
    integ = np.zeros((i_rows, i_cols), np.float)
    sqr_integral = np.zeros((i_rows, i_cols), np.float)

    integ[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float), axis=0), axis=1)
    sqr_img = np.square(img.astype(np.float))
    sqr_integral[1:, 1:] = np.cumsum(np.cumsum(sqr_img, axis=0), axis=1)

    # Defining grid
    x, y = np.meshgrid(np.arange(1, i_cols), np.arange(1, i_rows))

    # Obtaining local coordinates
    hw_size = w_size // 2
    x1 = (x - hw_size).clip(1, cols)
    x2 = (x + hw_size).clip(1, cols)
    y1 = (y - hw_size).clip(1, rows)
    y2 = (y + hw_size).clip(1, rows)

    # Obtaining local areas size
    l_size = (y2 - y1 + 1) * (x2 - x1 + 1)

    # Computing sums
    sums = integ[y2, x2] - integ[y2, x1 - 1] - integ[y1 - 1, x2] + integ[y1 - 1, x1 - 1]
    sqr_sums = (
        sqr_integral[y2, x2]
        - sqr_integral[y2, x1 - 1]
        - sqr_integral[y1 - 1, x2]
        + sqr_integral[y1 - 1, x1 - 1]
    )

    # Computing local means
    means = sums / l_size

    # Computing local standard deviation
    stds = np.sqrt(sqr_sums / l_size - np.square(means))

    # Computing thresholds
    thresholds = means * (1.0 + k * (stds / 128 - 1.0))

    return thresholds


# --------------------------------------------
# numpy(single) [0, 1] <--->  numpy(unit)
# --------------------------------------------
def uint2single(img):

    return np.float32(img / 255.0)


def single2uint(img):

    return np.uint8((img.clip(0, 1) * 255.0).round())


def uint162single(img):

    return np.float32(img / 65535.0)


def single2uint16(img):

    return np.uint16((img.clip(0, 1) * 65535.0).round())


def get_proc(processes):
    proc = ""
    for i in processes:
        if i != "+" and i != "|" and i != "\\" and i != "//" and i != "*":
            proc += i
        else:
            return proc
    return proc


def preprocess(image, processes):
    """
    Preprocess operations on image uses OpenCV.
    Usage: image = preprocess(image, 'CODE')
        | 1 - morphology
        | 2 - cleanCorners:   sets the borders of the image as bg
        | 3 - eraseTwoByTwos: detecting/reducing interest areas
        | 4 - eraseLadders:   used in Zhang Suen algorithms to eliminate undesired corners
        | G - get grayscale image
        | B - Median Blur
        | N - Gaussian Blur
        | L - Laplacian Gradient
        | T - Otsu thresholding
        | t - adaptive tresholding
        | D - dilation
        | E - erosion
        | O - opening - erosion followed by dilation
        | C - canny edge detection
        | S - scew correction
        | M - template matching
        | R - descew (autorotate) image
    """
    loginfo(f" | IMAGE: {image.shape} -> ")
    for j in range(len(processes)):
        i = get_proc(processes)
        loginfo(f" | IMAGE | DO PROCESSING: {i} ...")
        if "MORPHOLOGY" in i or "morphology" in i:
            image = morphology(image)
        if "CLEAN-CORNERS" in i or "clean-corners" in i:
            cleanCorners(image)
        if "ERASE-TWO-BY-TWOS" in i or "erase-two-by-twos" in i:
            eraseTwoByTwos(image)
        if "ERASE-LADDERS" in i or "erase-ladders" in i:
            eraseLadders(image)
        if "NICE-PROC" in i or "nice-proc" in i:
            image = nice_proc(image)
        if "KNN" in i or "knn" in i:
            backSub = cv2.createBackgroundSubtractorKNN()
            image = backSub.apply(image)
        if "MOG2" in i or "mog2" in i:
            backSub = cv2.createBackgroundSubtractorMOG2()
            image = backSub.apply(image)
        if "BGR2GRAY" in i or "bgr2gray" in i:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if "BGR2RGB" in i or "bgr2rgb" in i:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if "MEDIAN-BLUR" in i or "median-blur" in i:
            image = cv2.medianBlur(image, 3)
        if "GAUSSIAN-BLUR" in i or "gaussian-blur" in i:
            image = cv2.GaussianBlur(image, (3, 3), 1)
        if "BILATERIAL" in i or "bilateral" in i:
            image = cv2.bilateralFilter(image, 3, 10, 10)
        if "SHARPENING" in i or "sharpening" in i:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel)
        if "LAPLACIAN" in i or "laplacian" in i:
            image = cv2.Laplacian(image, cv2.CV_64F)
        if "THRESH-OTSU" in i or "thresh-otsu" in i:
            image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]
        if "ADAPTIVE-TRESH-1" in i or "adaptive-thresh-1" in i:
            image = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2
            )
        if "ADAPTIVE-TRESH-2" in i or "adaptive-thresh-2" in i:
            image = adaptative_thresholding(image, 1)
        if "SAUVOLA-TRESH" in i or "sauvola-thresh" in i:
            image = sauvola_threshold(image, w_size=3, k=0.2)
        if "DILATE" in i or "dilate" in i:
            kernel = np.ones((2, 2), np.uint8)
            image = cv2.dilate(image, kernel, iterations=1)
        if "ERODE" in i or "erode" in i:
            kernel = np.ones((2, 2), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)
        if "OPEN" in i or "ones" in i:
            kernel = np.ones((2, 2), np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        if "CLOSE" in i or "ones" in i:
            kernel = np.ones((2, 2), np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        if "CANNY" in i or "canny" in i:
            image = cv2.Canny(image, 100, 200)
        if "EQUALIZE-HIST" in i or "equalize-hist" in i:
            image = cv2.equalizeHist(image)
        if "DESCEW-1" in i or "descew-1" in i:
            coords = np.column_stack(np.where(image > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

        # Delete from DO PIPE alredy executed processings:
        try:
            processes = processes[len(i) + 1 :]
            if len(processes) == 0:
                break
        except IndexError:
            break

    if image.dtype != "uint8":
        image = single2uint(image)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.uint8(image)

    loginfo(f" | -> IMAGE: {image.shape}")
    return image


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    pts = np.array(pts)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def cropping_from_box(image, box):
    mask = np.zeros(image.shape, dtype=np.uint8)
    a = (int(box[0][0]), int(box[0][1]))
    b = (int(box[1][0]), int(box[1][1]))
    c = (int(box[2][0]), int(box[2][1]))
    d = (int(box[3][0]), int(box[3][1]))
    roi_corners = np.array([[a, b, c, d]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly (it's convex)
    # apply the masK
    masked_image = cv2.bitwise_and(image, mask)
    image = four_point_transform(masked_image, box)

    return np.uint8(image)


def cropping_image(image, boxes):
    """
    #-MAIN-DATA--------------------------------
        | input:    np.array    (H, W, C) | RGB
        | output:   [np.array, ...]    (H, W, C) | RGB
    #-ARGUMENTS--------------------------------
        | - image       np.array    - 3 dim image needed cropped
        | - boxes       [[[int, int], [int, int], [int, int], [int, int]], ...]
    #-USAGE------------------------------------
        | Use to cropping image on boxes cordinates.
        | return [np.array, np.array, ...]
    """
    loginfo(f" | BEGIN (IMAGE{image.shape}, BOXES LEN:{len(boxes)}) -> ")

    images_cropped = []
    for idx, box in enumerate(boxes):
        images_cropped.append(cropping_from_box(image, box))

    loginfo(f" | -> DONE")
    return images_cropped


def cropping_image_all(image_shape, boxes):
    """
    #-MAIN-DATA--------------------------------
        | input:    np.array    (H, W, C) | RGB
        | output:   np.array    (H, W, C) | RGB
    #-ARGUMENTS--------------------------------
        | - image       np.array    - 3 dim image needed cropped
        | - boxes       [[[int, int], [int, int], [int, int], [int, int]], ...]
    #-USAGE------------------------------------
        | Use to cropping image on bottom, apper, left, right boxes cordinates.
        | return    np.array
    """
    logging.info(f" | BEGIN (IMAGE{image_shape}, BOXES LEN:{len(boxes)}) -> ")

    (h, w, c) = image_shape
    box_cr = [[int(w / 2), int(h / 2)] for i in range(4)]

    DIFF = 50

    box_0_0 = np.min([box[0][0] for box in boxes])
    box_0_1 = np.min([box[0][1] for box in boxes])
    box_1_0 = np.max([box[1][0] for box in boxes])
    box_1_1 = np.min([box[1][1] for box in boxes])
    box_2_0 = np.max([box[2][0] for box in boxes])
    box_2_1 = np.max([box[2][1] for box in boxes])
    box_3_0 = np.min([box[3][0] for box in boxes])
    box_3_1 = np.max([box[3][1] for box in boxes])

    box_cr = [
        [box_0_0 - DIFF, box_0_1 - DIFF],
        [box_1_0 + DIFF, box_1_1 - DIFF],
        [box_2_0 + DIFF, box_2_1 + DIFF],
        [box_3_0 - DIFF, box_3_1 + DIFF],
    ]

    if box_cr[0][0] < 0:
        box_cr[0][0] = 0
    if box_cr[0][1] < 0:
        box_cr[0][1] = 0
    if box_cr[1][0] > (w - 1):
        box_cr[1][0] = w - 1
    if box_cr[1][1] < 0:
        box_cr[1][1] = 0
    if box_cr[2][0] > (w - 1):
        box_cr[2][0] = w - 1
    if box_cr[2][1] > (h - 1):
        box_cr[2][1] = h - 1
    if box_cr[3][0] < 0:
        box_cr[3][0] = 0
    if box_cr[3][1] > (h - 1):
        box_cr[3][1] = h - 1

    # Convert box cordinates:
    boxes = [
        (
            (box[0][0] - box_cr[0][0], box[0][1] - box_cr[0][1]),
            (box[1][0] - box_cr[0][0], box[1][1] - box_cr[0][1]),
            (box[2][0] - box_cr[0][0], box[2][1] - box_cr[0][1]),
            (box[3][0] - box_cr[0][0], box[3][1] - box_cr[0][1]),
        )
        for box in boxes
    ]

    w1 = box_cr[1][0] - box_cr[0][0]
    w2 = box_cr[2][0] - box_cr[0][0]
    w3 = box_cr[1][0] - box_cr[3][0]
    w4 = box_cr[2][0] - box_cr[3][0]
    w = int(np.max([w1, w2, w3, w4]))
    h1 = box_cr[2][1] - box_cr[0][1]
    h2 = box_cr[3][1] - box_cr[0][1]
    h3 = box_cr[2][1] - box_cr[1][1]
    h4 = box_cr[3][1] - box_cr[1][1]
    h = int(np.max([h1, h2, h3, h4]))
    cropped_image_shape = (h, w, c)

    logging.info(f" | -> DONE (IMAGE{cropped_image_shape}) ")
    return cropped_image_shape, boxes, box_cr


def rotate_box(box, angle, image_shape):
    for i in range(angle):
        if i % 2 == 0:
            (h, w, c) = image_shape
        if i % 2 == 1:
            (w, h, c) = image_shape
        box = (
            (box[1][1], w - box[1][0]),
            (box[2][1], w - box[2][0]),
            (box[3][1], w - box[3][0]),
            (box[0][1], w - box[0][0]),
        )
    return box


def rotate_boxes(boxes, angle, image_shape):
    loginfo(f" | IMAGE ANGLE: {angle * 90} -> ROTATING...")
    boxes = [rotate_box(box, angle, image_shape) for box in boxes]
    loginfo(f" | ROTATING DONE")
    return boxes


def count_vertical_boxes(boxes):
    count_boxes = 0
    for idx, box in enumerate(boxes):
        h_box = int(box[3][1] - box[0][1])
        w_box = int(box[1][0] - box[0][0])
        d = int(h_box - w_box)
        if d > 0:
            count_boxes += 1
    return count_boxes


def check_size(boxes, size_drop):
    """
    #-MAIN-DATA--------------------------------
        | input:    [BOX, BOX, BOX, ...] or [boxes, texts, scors]
        | output:   [BOX, BOX, BOX, ...] or [boxes, texts, scors]
    #-ARGUMENTS--------------------------------
        | - boxes       [[[int, int], [int, int], [int, int], [int, int]], ...]
        | - size_drop   int - if h or w of BOX < size_drop -> delete this BOX
    #-USAGE------------------------------------
        | Use to avoid very small BOXES.
        | return [BOX, BOX, ...]
    """
    try:
        if type(boxes[0][0][0]) == int or type(boxes[0][0][0]) == float:
            boxes_new = []
            for box in boxes:
                h = int(box[3][1] - box[0][1])
                w = int(box[1][0] - box[0][0])
                if not ((h < size_drop) or (w < size_drop)):
                    boxes_new.append(box)
            boxes = boxes_new
        else:
            boxes, texts, scors = boxes
            boxes_new = []
            texts_new = []
            scors_new = []
            for box, txt, scr in zip(boxes, texts, scors):
                h = int(box[3][1] - box[0][1])
                w = int(box[1][0] - box[0][0])
                if not ((h < size_drop) or (w < size_drop)):
                    boxes_new.append(box)
                    texts_new.append(txt)
                    scors_new.append(scr)
            boxes = [boxes_new, texts_new, scors_new]
    except AttributeError as e:
        return boxes
    return boxes


def get_data_image(data, image_number):
    """
    #-MAIN-DATA--------------------------------
        | input:    [[N_IMAGE_0, [[BOX], [TXT, SCR]], ...], ...  [[N_IMAGE_I, ...]]]
        | output:   [[[BOX], [TXT, SCR]], ... ]
    #-USAGE------------------------------------
        | Use to get all data for specific image.
    """
    data_image = []

    # Search all lines with data for image with specific number:
    for j in range(len(data)):
        if data[j][0] == image_number:
            # data[j]  =  [N_IMAGE, [[BOX], [TXT, SCR]], ...]
            data_ = data[j][1:]

            # block = [[BOX], [TXT, SCR]]
            for block in data_:
                data_image.append(block)

    return data_image


def check_data_image(result, image_shape=(1000, 1000, 3), blacklist=False):
    """
    #-MAIN-DATA--------------------------------
        | input:    [[[BOX], [TXT, SCR]], ... ]
        | output:   boxes = [BOX, BOX, BOX, ... ]
        |           texts = [TXT, TXT, TXT, ... ]
        |           scors = [SCR, SCR, SCR, ... ]
    #-USAGE------------------------------------
        | Use to check data from OCR, and avoid bad data.
    """
    (h, w, c) = image_shape

    boxes = []
    texts = []
    scors = []
    last_box = []

    # Watching each batch (or line) with format of batch [BOX, [TEXT, SCORE]]
    for line in result:

        # Single BOX
        box = line[0]

        # Check if we have box in batch. If not -> think that we have box in previos bathces.
        if box != 0:
            if len(box) != 0:
                # Checking if we have very small boxes:
                try:
                    box = check_size([box], int(h / 50))[0]
                except IndexError:
                    box = []
        elif box == 0:
            box = []

        if len(box) != 0:
            last_box = box

        # [TEXT, SCORE]
        texts_scores = line[1]
        # If we have not and TEXT and SCORES and have not last box (-> means have not box in place to) ->
        # -> Skip with batch.

        if len(last_box) == 0:
            continue

        if len(texts_scores) == 0:
            continue

        # Checking for existing text, if not, create empty text with 0 score:
        elif len(texts_scores) != 0:

            # Checking text with regular expression (avoid bad symbols):
            text_before = str(texts_scores[0])

            # Additional symbols, that needed recognized too:
            additional_symbols = r"d-.%s"
            _ = ""
            for s in additional_symbols:
                _ += "\\" + s
            additional_symbols = _

            symbols = "[^a-zA-Zа-яА-Я" + additional_symbols + "]+"

            if blacklist == True:
                symbols = "[^" + additional_symbols + "]+"

            text_re = re.sub(symbols, "", text_before)

            # If all ok, and we have data -> add last_box to boxes to drawing.
            if text_re != 0 and text_re != "":
                texts.append(str(text_re))
                boxes.append(last_box)
            else:
                continue

            # Try append SCORE to SCORES, converting to INT.
            try:
                scors.append(int(texts_scores[1]))
            except Exception as e:
                scors.append(70)
        else:
            texts.append("")
            scors.append(0)

    return boxes, texts, scors


def check_boxes_cordinates(data, result, image, i=0):
    # Ckeck if we alredy have boxes and convert new boxes to cordinates old_boxes (init image):
    if len(data) > 0:
        boxes_new = []
        # result =  [[BOX, []], ...]
        boxes = [line[0] for line in result]
        txtsc = [line[1] for line in result]

        # data[j]  =  [N_IMAGE, [[BOX], [TXT, SCR]], ...]
        box_old = data[i][1][0]

        # Nedded evaluate coeff of converting (image.shape = box_old or not):
        coeff = image.shape[1] / (box_old[0][0] + box_old[1][0])
        boxes = np.uint8(np.array(boxes) / coeff).tolist()

        # Convert box cordinates:
        for box in boxes:
            box_new = (
                (box_old[0][0] + box[0][0], box_old[0][1] + box[0][1]),
                (box_old[0][0] + box[1][0], box_old[0][1] + box[1][1]),
                (box_old[0][0] + box[2][0], box_old[0][1] + box[2][1]),
                (box_old[0][0] + box[3][0], box_old[0][1] + box[3][1]),
            )
            boxes_new.append(box_new)

        boxes = boxes_new

        # Create output in standart format [[BOX, [TXT, SCR]], ... ]:
        result_ = []
        for j, box in enumerate(boxes):
            result_.append([box, txtsc[j]])
        result = result_

    return result


def check_boxes_cordinates_ic(
    data, data_res, images_before, images_cutted, results_images, result, i=0
):
    for j, image_cutted in enumerate(images_cutted):
        results_images.append([images_before[0], image_cutted])

        # Ckeck if we alredy have boxes and convert new boxes to cordinates old_boxes (init image)
        if len(data) > 0:
            # Number of image:
            data_ = data[i][0]

            # result[j] = [BOX, []]  or  [BOX, [TEXT, SCORES]]
            box = result[j][0]
            txt = result[j][1]

            box_old = data[i][-1][0]

            # Nedded evaluate coeff of converting (image.shape = box_old or not):
            coeff = images_before[i].shape[1] / (box_old[0][0] + box_old[1][0])
            boxes = np.uint8(np.array(boxes) / coeff).tolist()

            box_new = (
                (box_old[0][0] + box[0][0], box_old[0][1] + box[0][1]),
                (box_old[0][0] + box[1][0], box_old[0][1] + box[1][1]),
                (box_old[0][0] + box[2][0], box_old[0][1] + box[2][1]),
                (box_old[0][0] + box[3][0], box_old[0][1] + box[3][1]),
            )

            # Converting BOX to common output:
            # OUT:  [NUMBER_OF_IMAGE, [BOX, []]]
            res = [data_, [box_new, txt]]
            data_res.append(res)
        else:
            # OUT:  [NUMBER_OF_IMAGE, [BOX, []]]
            data_res.append([images_before[0], result[j]])
