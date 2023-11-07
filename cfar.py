import numpy as np


def ca_cfar(range_doppler_map, guard_cells, training_cells, threshold_factor):
    """
    Perform 2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) detection on a range-Doppler map.

    Args:
        range_doppler_map (numpy.ndarray): 2D range-Doppler map.
        guard_cells (tuple): Number of guard cells in both range and Doppler dimensions (e.g., (4, 4)).
        training_cells (tuple): Number of training cells in both range and Doppler dimensions (e.g., (8, 8)).
        threshold_factor (float): Threshold factor to determine the detection threshold (e.g., 1.5).

    Returns:
        detection_map (numpy.ndarray): Binary detection map (1 for detected targets, 0 for clutter).
    """
    num_range_bins, num_doppler_bins = range_doppler_map.shape
    detection_map = np.zeros_like(range_doppler_map, dtype=int)

    for r in range(guard_cells[0], num_range_bins - guard_cells[0]):
        for d in range(guard_cells[1], num_doppler_bins - guard_cells[1]):
            # Create local region for training and guard cells
            training_cells_sum = np.sum(range_doppler_map[r - guard_cells[0]:r + guard_cells[0] + 1,
                                        d - guard_cells[1]:d + guard_cells[1] + 1])
            guard_cells_sum = np.sum(range_doppler_map[r - guard_cells[0]:r + guard_cells[0] + 1,
                                     d - guard_cells[1]:d + guard_cells[1] + 1])

            # Calculate the average noise level in the training cells
            training_cells_avg = (training_cells_sum - guard_cells_sum) / (
                    (training_cells[0] * training_cells[1]) - (guard_cells[0] * guard_cells[1]))

            # Calculate the threshold
            threshold = threshold_factor * training_cells_avg
            print(threshold)

            # If the cell value is above the threshold, mark it as a detection
            if range_doppler_map[r, d] > threshold:
                detection_map[r, d] = 1


    print(np.max(detection_map))
    return np.where(detection_map, range_doppler_map, 0.0)