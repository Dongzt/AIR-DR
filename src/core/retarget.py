import cv2
import numpy as np

class Instance:
    def __init__(self, instance_id, bbox, weight, mask, true_area):
        self.instance_id = instance_id
        self.bbox = bbox
        self.weight = weight
        self.mask = mask
        self.true_area = true_area

def calculate_area_retention(image, instances):
    total_weighted_ratio = 0.0
    for instance in instances:
        target_value = instance.instance_id
        current_pixels = np.sum(image == target_value)
        retention_ratio = current_pixels / instance.true_area if instance.true_area > 0 else 0
        total_weighted_ratio += retention_ratio * instance.weight
    return total_weighted_ratio 

def filter_bounding_boxes(image, min_area):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    instances = []
    valid_contours = []
    total_bbox_area = 0
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        
        instance_id = max(0, 255 - i * 20 - 1)
        mask_roi = np.zeros((h, w), dtype=np.uint8)
        offset_contour = contour - np.array([x, y])
        cv2.drawContours(mask_roi, [offset_contour], -1, 255, cv2.FILLED)
        true_area = cv2.countNonZero(mask_roi)
        
        valid_contours.append({
            "id": instance_id,
            "bbox": (x, y, w, h),
            "mask": mask_roi,
            "bbox_area": w * h,
            "true_area": true_area
        })
        total_bbox_area += w * h
        cv2.drawContours(image, [contour], -1, color=instance_id, thickness=cv2.FILLED)

    for contour_info in valid_contours:
        weight = contour_info["bbox_area"] / total_bbox_area if total_bbox_area > 0 else 0.0
        instances.append(Instance(
            instance_id=contour_info["id"],
            bbox=contour_info["bbox"],
            weight=weight,
            mask=contour_info["mask"],
            true_area=contour_info["true_area"]
        ))

    instances.sort(key=lambda x: (x.bbox[1] + x.bbox[3]/2))
    return instances

def crop_output_image(origin_w, origin_h, output_image, target_ratio):
    contours, _ = cv2.findContours(output_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return output_image, 0, 0

    x_coords = [point[0][0] for contour in contours for point in contour]
    y_coords = [point[0][1] for contour in contours for point in contour]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    instance_width = max_x - min_x
    instance_height = max_y - min_y

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
        return output_image, 0, 0

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    crop_x1 = max(0, center_x - crop_width // 2)
    crop_y1 = max(0, center_y - crop_height // 2)
    crop_x2 = min(output_image.shape[1], crop_x1 + crop_width)
    crop_y2 = min(output_image.shape[0], crop_y1 + crop_height)

    crop_x1 = max(0, crop_x2 - crop_width)
    crop_y1 = max(0, crop_y2 - crop_height)

    cropped_image = output_image[crop_y1:crop_y2, crop_x1:crop_x2]
    return cropped_image, crop_x1, crop_y1

def place_instance(origin_w, origin_h, output_image, instance, bbox, new_w, new_h, displacement=0):
    x, y, w_box, h_box = bbox
    center_x = x + w_box // 2
    
    new_center_x = int(center_x * new_w / origin_w)

    absolute_displacement = int(displacement * 300)
    new_x = max(0, min(new_center_x - w_box // 2 + absolute_displacement, new_w - w_box))
    new_y = max(0, min(output_image.shape[0]// 2 - origin_h // 2 + y , new_h - h_box))
    
    target_roi = output_image[new_y:new_y + h_box, new_x:new_x + w_box]
    instance_mask = (instance > 0).astype(np.uint8)

    new_instance = cv2.bitwise_and(instance, instance, mask=instance_mask)
    target_roi = np.where(new_instance > 0, new_instance, target_roi)

    output_image[new_y:new_y + h_box, new_x:new_x + w_box] = target_roi

    return output_image , new_x, new_y

def single_process_function(image, mask, target_ratio, bbox):
    x, y, w, h = bbox
    ori_h, ori_w = image.shape[:2]
    original_ratio = ori_h / ori_w

    paint_flag = True
    epsilon = 0.3
    
    if original_ratio < target_ratio:
        if w * target_ratio > ori_h and abs(original_ratio-target_ratio) > epsilon:
            new_w = w
            new_h = int(new_w * target_ratio)
        else:
            new_h = h
            new_w = int(new_h // target_ratio)
            paint_flag = False
        if (w * h)/(ori_h * ori_w) < 0.2:
            new_h = ori_h
            new_w = int(new_h // target_ratio)
    else:
        if h // target_ratio > ori_w and abs(original_ratio-target_ratio) > epsilon:
            new_h = h
            new_w = int(new_h // target_ratio)
        else:
            new_w = w
            new_h = int(new_w * target_ratio)
            paint_flag = False
        if (w * h)/(ori_h * ori_w) < 0.2:
            new_w = ori_w
            new_h = int(new_w * target_ratio)
    
    if new_h < h or new_w < w:
        paint_flag = True

    if paint_flag:
        paint_bg = np.full((new_h, new_w, 3), 128, dtype=np.uint8)
        x_center = x + w // 2
        y_center = y + h // 2
        
        crop_x1 = max(0, int(x_center - new_w // 2))  
        crop_y1 = max(0, int(y_center - new_h // 2))  
        crop_x2 = min(ori_w, int(crop_x1 + new_w))    
        crop_y2 = min(ori_h, int(crop_y1 + new_h))    
        
        if crop_x2 - crop_x1 > new_w:
            if crop_x1 == 0: crop_x2 = new_w
            else: crop_x1 = crop_x2 - new_w
        if crop_y2 - crop_y1 > new_h:
            if crop_y1 == 0: crop_y2 = new_h
            else: crop_y1 = crop_y2 - new_h
            
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        c_h, c_w = cropped.shape[:2]
        
        paste_x = (new_w - c_w) // 2
        paste_y = (new_h - c_h) // 2
        paste_x = max(0, paste_x)
        paste_y = max(0, paste_y)
        
        paint_bg[paste_y:paste_y+c_h, paste_x:paste_x+c_w] = cropped
        
        paint_mask = np.full((new_h, new_w), 255, dtype=np.uint8)
        paint_mask[paste_y:paste_y+c_h, paste_x:paste_x+c_w] = 0

        return paint_bg, paint_mask, paint_flag
    else:
        x_center = x + w // 2
        y_center = y + h // 2

        crop_x1 = max(0, int(x_center - new_w // 2))  
        crop_y1 = max(0, int(y_center - new_h // 2))  
        crop_x2 = min(ori_w, int(crop_x1 + new_w))    
        crop_y2 = min(ori_h, int(crop_y1 + new_h))    

        if crop_x2 - crop_x1 < new_w:
            if crop_x1 == 0: crop_x2 = new_w
            else: crop_x1 = crop_x2 - new_w
        if crop_y2 - crop_y1 < new_h:
            if crop_y1 == 0: crop_y2 = new_h
            else: crop_y1 = crop_y2 - new_h
        
        paint_bg = image[crop_y1:crop_y2, crop_x1:crop_x2]
        paint_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        return paint_bg, paint_mask, paint_flag

def add_bg(mask, image, retarget_mask, retarget_image):
    new_retarget_mask = retarget_mask.copy()
    ori_h, ori_w = mask.shape
    target_h, target_w = retarget_mask.shape
    directions = ["top", "bottom", "left", "right"]
    
    instances = np.unique(mask)
    instances = instances[instances != 0]
    
    y_min, y_max, x_min, x_max = ori_h, 0, ori_w, 0
    y_min_t, y_max_t, x_min_t, x_max_t = target_h, 0, target_w, 0
    
    for instance_id in instances:
        orig_mask = (mask == instance_id).astype(np.uint8)
        orig_contours, _ = cv2.findContours(orig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not orig_contours:
            continue
        x_orig, y_orig, w_orig, h_orig = cv2.boundingRect(max(orig_contours, key=cv2.contourArea))
        y_min = min(y_min, y_orig)
        y_max = max(y_max, y_orig+h_orig)
        x_min = min(x_min, x_orig)
        x_max = max(x_max, x_orig+w_orig)
        
        target_mask = (retarget_mask == instance_id).astype(np.uint8)
        target_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not target_contours:
            continue
        x_target, y_target, w_target, h_target = cv2.boundingRect(max(target_contours, key=cv2.contourArea))
        y_min_t = min(y_min_t, y_target)
        y_max_t = max(y_max_t, y_target + h_target)
        x_min_t = min(x_min_t, x_target)
        x_max_t = max(x_max_t, x_target + w_target)

    up = y_min
    down = ori_h - y_max
    left = x_min
    right = ori_w - x_max
    up_target = y_min_t
    down_target = target_h - y_max_t
    left_target = x_min_t
    right_target = target_w - x_max_t
    
    alpha = 0.3
    gap_y = ori_h * alpha
    gap_x = ori_w * alpha
    gap_y_target = target_h * alpha
    gap_x_target = target_w * alpha

    for idx, direction in enumerate(directions):
        region = None
        target_region = None
        
        if direction == "top" and target_h / target_w >= 1:
            if up > gap_y and up_target > gap_y_target:
                region = (0, 0, ori_w, up)
                target_region = (0, 0, target_w, up_target)
        elif direction == "bottom" and target_h / target_w >= 1:
            if down > gap_y and down_target > gap_y_target:
                region = (0, ori_h-down, ori_w, down)
                target_region = (0, target_h-down_target, target_w, down_target)
        elif direction == "left" and target_h / target_w < 1:
            if left > gap_x and left_target > gap_x_target:
                region = (0, 0, left, ori_h)
                target_region = (0, 0, left_target, target_h)
        elif direction == "right" and target_h / target_w < 1:
            if right > gap_x and right_target > gap_x_target:
                region = (ori_w-right, 0, right, ori_h)
                target_region = (target_w-right_target, 0, right_target, target_h)
        
        if region is not None:
            x, y, w, h = region
            rect_w = int(w * 0.8)  
            rect_h = int(h * 0.8)
            cx = x + (w - rect_w) // 2
            cy = y + (h - rect_h) // 2  
            
            x_target, y_target, w_target, h_target = target_region
            rect_w_target = int(w_target * 0.8)
            rect_h_target = int(h_target * 0.8)
            cx_target = x_target + (w_target - rect_w_target) // 2
            cy_target = y_target + (h_target - rect_h_target) // 2
            
            cv2.rectangle(new_retarget_mask, 
                        (cx_target, cy_target),
                        (cx_target + rect_w_target, cy_target + rect_h_target),
                        90 - idx * 10, -1)
            bg = image[cy:cy + rect_h, cx:cx + rect_w]
            resized = cv2.resize(bg, (rect_w_target, rect_h_target))
            retarget_image[cy_target:cy_target + rect_h_target, cx_target:cx_target + rect_w_target] = resized
    
    return retarget_image, new_retarget_mask

def generate_layout(image, target_ratio, instances, displacements):
    h, w = image.shape
    original_ratio = h / w
    if_resize = True
    
    if original_ratio > 1 and target_ratio > 1:
        if_resize = False
        new_w = int(h / target_ratio)
        new_h = h
    elif original_ratio < 1 and target_ratio < 1:
        if_resize = False
        new_w = w
        new_h = int(w * target_ratio)
    elif original_ratio >= 1 and target_ratio <= 1:
        new_w = w
        new_h = int(w * target_ratio)
    else:
        new_w = int(h / target_ratio)
        new_h = h

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
    for instance in instances:
        x, y, w_box, h_box = instance.bbox
        
        if w_box > new_w:
            if_resize = False
            new_w = w_box
            new_h = int(new_w * target_ratio) 
        if h_box > new_h:
            if_resize = False
            new_h = h_box
            new_w = int(new_h / target_ratio)

    old_area = -1
    new_area = 0
    gamma = 0.95

    while True:
        output_image = np.zeros((new_h, new_w), dtype=np.uint8)
        target_x = []
        target_y = []
        for instance in instances:
            instance_mask = instance.mask
            bbox = instance.bbox
            x, y, w_box, h_box = bbox
            tmp_instance = image[y:y + h_box, x:x + w_box]
            masked_instance = cv2.bitwise_and(tmp_instance, tmp_instance, mask=instance_mask)
            output_image, new_x, new_y = place_instance(image.shape[1], image.shape[0], output_image, masked_instance, bbox, new_w, new_h, 0)
            target_x.append(new_x)
            target_y.append(new_y)
        area_retention = calculate_area_retention(output_image, instances)

        if area_retention >= gamma:
            break
        old_area = new_area
        new_area = area_retention
        if old_area == new_area:
            break
        
        new_w = int(new_w * 1.25)
        new_h = int(new_w * target_ratio)

    if if_resize and len(instances) > 1:
        output_image = np.zeros((new_h, new_w), dtype=np.uint8)
        target_x = []
        target_y = []
        for i, instance in enumerate(instances):
            instance_mask = instance.mask
            bbox = instance.bbox
            x, y, w_box, h_box = bbox
            tmp_instance = image[y:y + h_box, x:x + w_box]
            masked_instance = cv2.bitwise_and(tmp_instance, tmp_instance, mask=instance_mask)
            
            displacement_val = displacements[i] if i < len(displacements) else 0
            output_image, new_x, new_y = place_instance(image.shape[1], image.shape[0], output_image, masked_instance, bbox, new_w, new_h, displacement_val)
            target_x.append(new_x)
            target_y.append(new_y)

        area_retention = calculate_area_retention(output_image, instances)
        output_image, crop_x1, crop_y1 = crop_output_image(image.shape[1], image.shape[0], output_image, target_ratio)

        target_x = [x - crop_x1 for x in target_x]
        target_y = [y - crop_y1 for y in target_y]
    else:
        output_image = np.zeros((new_h, new_w), dtype=np.uint8)
        target_x = []
        target_y = []
        for i, instance in enumerate(instances):
            instance_mask = instance.mask
            bbox = instance.bbox
            x, y, w_box, h_box = bbox
            tmp_instance = image[y:y + h_box, x:x + w_box]

            if len(instances) == 1:
                modified_mask = np.ones_like(instance_mask)
                masked_instance = cv2.bitwise_and(tmp_instance, tmp_instance, mask=modified_mask)
            else:
                masked_instance = cv2.bitwise_and(tmp_instance, tmp_instance, mask=instance_mask)
            
            output_image, new_x, new_y= place_instance(image.shape[1], image.shape[0], output_image, masked_instance, bbox, new_w, new_h, 0)
            target_x.append(new_x)
            target_y.append(new_y)

        area_retention = calculate_area_retention(output_image, instances)

    return output_image, target_x, target_y, if_resize

def create_inpaint_map(rgb_image, mask, retarget_mask, target_x, target_y):
    h, w = retarget_mask.shape
    retargeted_foreground = np.full((h, w, 3), 128, dtype=rgb_image.dtype)
    
    instances = np.unique(mask)
    instances = instances[(instances != 0) & (instances != 255)]

    instance_info = []
    for instance_id in instances:
        orig_mask = (mask == instance_id).astype(np.uint8)
        orig_contours, _ = cv2.findContours(orig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not orig_contours:
            continue
        orig_contour = max(orig_contours, key=cv2.contourArea)
        x_orig, y_orig, w_orig, h_orig = cv2.boundingRect(orig_contour)
        center_y = y_orig + h_orig / 2
        instance_info.append((instance_id, center_y, x_orig, y_orig, w_orig, h_orig))
    
    instance_info.sort(key=lambda x: x[1])

    for idx, (info, x, y) in enumerate(zip(instance_info, target_x, target_y)):
        instance_id, _, x_orig, y_orig, w_orig, h_orig = info
        
        target_mask = (retarget_mask == instance_id).astype(np.uint8)
        instance_mask = (mask == instance_id).astype(np.uint8)

        target_area = np.count_nonzero(target_mask)
        instance_area = np.count_nonzero(instance_mask)

        if instance_area == 0 or target_area / instance_area < 0.3:
            continue

        instance_mask = instance_mask[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]
        instance_patch = rgb_image[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]
        
        placement_region = instance_mask > 0

        placement_x = x
        placement_y = y

        start_x = max(0, placement_x)
        end_x = min(w, placement_x + w_orig)
        start_y = max(0, placement_y)
        end_y = min(h, placement_y + h_orig)

        crop_x_start = start_x - placement_x
        crop_x_end = crop_x_start + (end_x - start_x)
        crop_y_start = start_y - placement_y
        crop_y_end = crop_y_start + (end_y - start_y)

        cropped_instance = instance_patch[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        cropped_mask = placement_region[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        
        retargeted_foreground[start_y:end_y, start_x:end_x][cropped_mask] = cropped_instance[cropped_mask]
        
    return retargeted_foreground