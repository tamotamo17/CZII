"""
Derived from:
https://github.com/cellcanvas/album-catalog/blob/main/solutions/copick/compare-picks/solution.py
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import cupy as cp
from cupyx.scipy import ndimage as cnd
import scipy.ndimage as ndi
from skimage.morphology import ball
from skimage.segmentation import watershed
from skimage.measure import regionprops


class ParticipantVisibleError(Exception):
    pass


def compute_metrics(reference_points, reference_radius, candidate_points):
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    # Prevent submitting multiple matches per particle.
    # This won't be be strictly correct in the (extremely rare) case where true particles
    # are very close to each other.
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn


def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        distance_multiplier: float,
        beta: int) -> float:
    '''
    F_beta
      - a true positive occurs when
         - (a) the predicted location is within a threshold of the particle radius, and
         - (b) the correct `particle_type` is specified
      - raw results (TP, FP, FN) are aggregated across all experiments for each particle type
      - f_beta is calculated for each particle type
      - individual f_beta scores are weighted by particle type for final score
    '''

    particle_radius = {
        'apo-ferritin': 60,
        'beta-amylase': 65,
        'beta-galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus-like-particle': 135,
    }

    weights = {
        'apo-ferritin': 1,
        'beta-amylase': 0,
        'beta-galactosidase': 2,
        'ribosome': 1,
        'thyroglobulin': 2,
        'virus-like-particle': 1,
    }

    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}

    # Filter submission to only contain experiments found in the solution split
    split_experiments = set(solution['experiment'].unique())
    submission = submission.loc[submission['experiment'].isin(split_experiments)]

    # Only allow known particle types
    if not set(submission['particle_type'].unique()).issubset(set(weights.keys())):
        raise ParticipantVisibleError('Unrecognized `particle_type`.')

    assert solution.duplicated(subset=['experiment', 'x', 'y', 'z']).sum() == 0
    assert particle_radius.keys() == weights.keys()

    results = {}
    for particle_type in solution['particle_type'].unique():
        results[particle_type] = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
        }

    for experiment in split_experiments:
        for particle_type in solution['particle_type'].unique():
            reference_radius = particle_radius[particle_type]
            select = (solution['experiment'] == experiment) & (solution['particle_type'] == particle_type)
            reference_points = solution.loc[select, ['x', 'y', 'z']].values

            select = (submission['experiment'] == experiment) & (submission['particle_type'] == particle_type)
            candidate_points = submission.loc[select, ['x', 'y', 'z']].values

            if len(reference_points) == 0:
                reference_points = np.array([])
                reference_radius = 1

            if len(candidate_points) == 0:
                candidate_points = np.array([])

            tp, fp, fn = compute_metrics(reference_points, reference_radius, candidate_points)

            results[particle_type]['total_tp'] += tp
            results[particle_type]['total_fp'] += fp
            results[particle_type]['total_fn'] += fn

    aggregate_fbeta = 0.0
    for particle_type, totals in results.items():
        tp = totals['total_tp']
        fp = totals['total_fp']
        fn = totals['total_fn']

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        aggregate_fbeta += fbeta * weights.get(particle_type, 1.0)

    if weights:
        aggregate_fbeta = aggregate_fbeta / sum(weights.values())
    else:
        aggregate_fbeta = aggregate_fbeta / len(results)
    return aggregate_fbeta


def fbeta_score_multiclass(y_true, y_pred, beta=1.0, num_classes=None, weights=[0,1,0,2,1,2,1]):
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1

    fbeta_scores = []
    class_names = ['background', 'apo-ferritin', 'beta-amylase', 'beta-galactosidase',
                   'ribosome', 'thyroglobulin', 'virus-like-particle',
                   ]
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            fbeta = 0.0
        else:
            beta_sq = beta ** 2
            fbeta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
        fbeta_scores.append(fbeta*weights[cls])

    return np.mean(fbeta_scores)


def coords_from_segmentation_gpu(segmentation, segmentation_idx, maxima_filter_size, min_particle_size, max_particle_size, voxel_spacing_xy=1, voxel_spacing_z=1):
    """
    Process a specific label in the segmentation, extract centroids using GPU-accelerated preprocessing,
    perform watershed on CPU, and return centroids.

    Args:
        segmentation (np.ndarray): Multilabel segmentation array, shape=(num_slices, height, width)
        segmentation_idx (int): The specific label from the segmentation to process
        maxima_filter_size (int): Size of the maximum detection filter
        min_particle_size (int): Minimum size threshold for particles
        max_particle_size (int): Maximum size threshold for particles
        voxel_spacing (int): The voxel spacing used to scale pick locations (default 1)

    Returns:
        list of dict: List of centroids with scaled 'x', 'y', 'z' coordinates
    """
    # Transfer segmentation to GPU
    segmentation_gpu = cp.asarray(segmentation)

    # Create a binary mask for the specific segmentation label
    binary_mask_gpu = (segmentation_gpu == segmentation_idx).astype(cp.int32)

    # Skip if the segmentation label is not present
    if cp.sum(binary_mask_gpu) == 0:
        print(f"No segmentation with label {segmentation_idx} found.")
        return []

    # Structuring element for erosion and dilation (3D ball)
    struct_elem = ball(1).astype(np.int32)
    struct_elem_gpu = cp.asarray(struct_elem)

    # Perform binary erosion and dilation on GPU
    eroded_gpu = cnd.binary_erosion(binary_mask_gpu, structure=struct_elem_gpu)
    dilated_gpu = cnd.binary_dilation(eroded_gpu, structure=struct_elem_gpu)

    # Distance transform on GPU
    distance_gpu = cnd.distance_transform_edt(dilated_gpu)

    # Local maxima detection on GPU
    # Convert distance to CPU for peak_local_max since it's not GPU-compatible
    distance_cpu = cp.asnumpy(distance_gpu)
    local_max = (distance_cpu == ndi.maximum_filter(distance_cpu, footprint=np.ones((maxima_filter_size, maxima_filter_size, maxima_filter_size))))
    #local_max = peak_local_max(distance_cpu, indices=False, footprint=np.ones((maxima_filter_size, maxima_filter_size, maxima_filter_size)), labels=cp.asnumpy(dilated_gpu))

    # Labeling local maxima on CPU
    markers, num_features = ndi.label(local_max)

    # Watershed segmentation on CPU
    watershed_labels = watershed(-distance_cpu, markers, mask=cp.asnumpy(dilated_gpu))

    # Extract region properties and filter based on particle size
    regions = regionprops(watershed_labels)
    all_centroids = []
    for region in regions:
        if min_particle_size <= region.area <= max_particle_size:
            centroid = region.centroid  # (z, y, x)
            all_centroids.append({
                'x': centroid[2] * voxel_spacing_xy,
                'y': centroid[1] * voxel_spacing_xy,
                'z': centroid[0] * voxel_spacing_z
            })

    return all_centroids