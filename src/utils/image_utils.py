import cv2
import numpy as np
from PIL import Image

def numpy_to_pil(image, mode=None):
    """
    通用转换函数: Numpy (H,W,C) -> PIL Image
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy.ndarray.")

    # 确保是 uint8
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype("uint8")

    if image.ndim == 2:
        return Image.fromarray(image, mode="L")
    elif image.ndim == 3:
        # BGR to RGB (OpenCV default)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image, mode=mode or "RGB")
    return Image.fromarray(image)

def compute_target_size(ratio, base_area=1024*1024, multiple=16):
    """根据比例和基准面积计算目标宽高，确保是 multiple 的倍数"""
    width = int((base_area / ratio) ** 0.5 // multiple * multiple)
    height = int((base_area * ratio) ** 0.5 // multiple * multiple)
    return height, width

def crop_center(image, target_ratio):
    """中心裁剪 (作为 fallback 方案)"""
    h, w = image.shape[:2]
    curr_ratio = h / w
    
    if curr_ratio > target_ratio:
        new_h = int(w * target_ratio)
        top = (h - new_h) // 2
        return image[top:top+new_h, :]
    else:
        new_w = int(h / target_ratio)
        left = (w - new_w) // 2
        return image[:, left:left+new_w]

def smart_crop_by_content(output_image, target_ratio):
    """
    原 crop_output_image: 根据非零区域裁剪，并保持比例
    """
    contours, _ = cv2.findContours(output_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return output_image, 0, 0

    x_coords = [point[0][0] for contour in contours for point in contour]
    y_coords = [point[0][1] for contour in contours for point in contour]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # 实例区域的宽度和高度
    instance_width = max_x - min_x
    instance_height = max_y - min_y

    # 根据目标比例计算裁剪区域
    crop_width = instance_width
    crop_height = int(crop_width * target_ratio)

    if crop_height < instance_height:
        crop_height = instance_height
        crop_width = int(crop_height / target_ratio)

    if origin_h < origin_w:
        crop_height = max(crop_height, origin_h)
        crop_width = int(crop_height / target_ratio)
    else:
        crop_width = max(crop_width, origin_w)
        crop_height = int(crop_width * target_ratio)

    if crop_width >= output_image.shape[1]:
        #print("不裁剪")
        return output_image, 0, 0
    # 计算裁剪框的左上角坐标和右下角坐标
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    crop_x1 = max(0, center_x - crop_width // 2)
    crop_y1 = max(0, center_y - crop_height // 2)
    crop_x2 = min(output_image.shape[1], crop_x1 + crop_width)
    crop_y2 = min(output_image.shape[0], crop_y1 + crop_height)

    # 确保裁剪框不超出图像边界
    crop_x1 = max(0, crop_x2 - crop_width)
    crop_y1 = max(0, crop_y2 - crop_height)

    # 裁剪图像
    cropped_image = output_image[crop_y1:crop_y2, crop_x1:crop_x2]
    #print("Original Image Shape:",output_image.shape)
    #print(f"Cropped Image Shape: {cropped_image.shape}")
    #print("裁剪")

    return cropped_image, crop_x1, crop_y1

def compute_target_size(ratio, base_area=1024*1024):
    """计算符合比例且面积接近 base_area 的宽高 (16的倍数)"""
    width = int((base_area / ratio) ** 0.5 // 16 * 16)
    height = int((base_area * ratio) ** 0.5 // 16 * 16)
    return height, width