import numpy as np

def calculate_salient_width(mask):

    if mask is None:
        return 0

    binary = (mask > 0).astype(np.uint8)

    return np.sum(np.any(binary, axis=0))

def compute_sdr_score(mask_ori, mask_retarget):

    w_ori = calculate_salient_width(mask_ori)
    w_out = calculate_salient_width(mask_retarget)
    
    if w_ori == 0:
        return None  
    
    if w_out == 0:
        return 1.0 
        
    sdr = abs(w_ori - w_out) / w_ori
    return sdr