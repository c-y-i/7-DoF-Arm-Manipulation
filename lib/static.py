import numpy as np
from math import pi

def filter_block_detections(all_detections, num_samples=5, detection_threshold=0.8):
    """
    Filter and median-average block detections.
    all_detections: list of lists of dict with keys ['id', 'position']
    """
    valid_blocks = {}
    for sample in all_detections:
        for block in sample:
            block_id = block['id']
            if block_id not in valid_blocks:
                valid_blocks[block_id] = []
            valid_blocks[block_id].append(block['position'])
    
    filtered_blocks = []
    for block_id, positions in valid_blocks.items():
        if len(positions) >= num_samples * detection_threshold:
            positions = np.array(positions)
            median_pos = np.median(positions, axis=0)
            filtered_blocks.append({
                'id': block_id,
                'position': median_pos,
                'confidence': len(positions) / num_samples
            })
    
    return sorted(filtered_blocks, key=lambda x: x['confidence'], reverse=True)

def get_pick_place_pose_params(block_position, stack_height, place_target):
    """
    Return a dictionary of pose parameters for each step in the pick/place sequence.
    We only return position and orientation arrays here.

    block_position: [x, y, z] of the block
    place_target: [px, py, pz] base target; stack_height is used to adjust vertical placement
    orientation: always [0, pi, pi] as per original code
    """
    orientation = [0, pi, pi]
    bx, by, bz = block_position
    px, py, pz = place_target

    return {
        'pre_grasp': ([bx, by, bz + 0.15], orientation),
        'grasp': ([bx, by, bz + 0.01], orientation),
        'lift': ([bx, by, bz + 0.15], orientation),
        'pre_place': ([px, py, stack_height + 0.1], orientation),
        'place': ([px, py, stack_height - 0.03], orientation),
        'retreat': ([px, py, stack_height + 0.15], orientation)
    }
