import cv2
import numpy as np
import scipy
import scipy.ndimage

############################################################
#  MASK Process
############################################################

def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
        mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
        mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = scipy.misc.imresize(
        mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask
