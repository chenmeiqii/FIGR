import sys
import os
import re
import json
import base64
from io import BytesIO
from PIL import Image
import argparse
def encode_image(image):
    """
    Convert a PIL.Image object or image file path to base64-encoded string, and get resolution info.
    
    Args:
        image: Can be a PIL.Image object or image file path.
    Returns:
        dict with keys:
        - 'base64': base64-encoded string
        - 'width': width in pixels
        - 'height': height in pixels
        - 'resolution': string "widthxheight"
    """
    img_obj = None

    if isinstance(image, str):
        # Handle file path
        img_obj = Image.open(image)
        with open(image, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # Handle PIL.Image object
        image = image.convert("RGB")
        img_obj = image
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    width, height = img_obj.size
    
    return {
        'base64': base64_str,
        'width': width,
        'height': height
    }