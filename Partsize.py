"""
Partsize.py — Particle Size Measurement from Grindometer Images
================================================================

This script implements an automated particle size analysis according to
DIN-standard grindometer measurement procedures ("Messvorschrift").

Physical setup:
    A grindometer is a metal block with a wedge-shaped groove that goes
    from deep (50 µm) at one end to zero depth at the other, over a
    length of 145 mm. A paint or paste sample is drawn along the groove.
    Where particles are larger than the groove depth, they create visible
    streaks ("Streifen"). The position of the first streak and the density
    of streaks indicate the maximum and mean particle sizes.

Processing pipeline:
    1. Load a probe image (JPEG photograph of the grindometer)
    2. Convert to HSV and apply thresholding to find the sample region
    3. Detect the sample contour and apply perspective correction
    4. Extract the Region of Interest (ROI) — the measurement groove
    5. Segment streaks vs. background using a Gaussian Mixture Model (GMM)
    6. Slice the image vertically and count streaks per 1 mm slice
    7. Map pixel positions to particle sizes in micrometers
    8. Report maximum particle size (first streak) and mean fineness (DIN)

Output:
    - GMM segmentation image  → images/GMM_Segmentierung_Probe{N}.png
    - Streak distribution plot → images/Streifenverteilung_Probe{N}.png
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

# =============================================================================
# CONFIGURATION — Physical and image processing parameters
# =============================================================================

# Which probe (sample) to analyze — change this to process a different image
PROBE_NUMBER = "5022"

# Paths
DATA_FOLDER = "./data/"
OUTPUT_FOLDER = "./images/"

# Physical dimensions of the grindometer
GROOVE_LENGTH_MM = 145.0       # Total length of the measurement groove in mm
MAX_PARTICLE_SIZE_UM = 50      # Groove depth at the deep end in micrometers

# Region of Interest (ROI) in the rectified 1500x3000 image
# These pixel coordinates define the measurement groove area after
# perspective correction and resizing to 1500x3000
RECTIFIED_IMAGE_SIZE = (1500, 3000)  # (width, height)
ROI_X_START = 650   # Left edge of measurement groove
ROI_X_END = 845     # Right edge of measurement groove
ROI_Y_START = 258   # Top of measurement groove (deep end, 50 µm)
ROI_Y_END = 2700    # Bottom of measurement groove (shallow end, 0 µm)

# Contour filtering: only keep contours that are reasonably large
# (to exclude noise) but smaller than the full image (to exclude the border)
MIN_CONTOUR_AREA = 100 * 500   # Minimum area in pixels²
MAX_CONTOUR_FRACTION = 0.9     # Maximum fraction of total image pixels

# Streak detection parameters
SLICE_HEIGHT_MM = 1            # Analyze in 1 mm vertical slices
MIN_STREAK_WIDTH_PX = 10       # Ignore streaks narrower than this (noise)
ADAPTIVE_THRESH_BLOCK_SIZE = 11  # Block size for adaptive thresholding
ADAPTIVE_THRESH_C = 2          # Constant subtracted from mean in adaptive threshold

# DIN standard: mean particle size is the position of the first 3 mm
# interval that contains between 5 and 10 streaks
DIN_INTERVAL_MM = 3            # Length of the DIN evaluation window
DIN_MIN_STREAKS = 5            # Minimum streaks in the DIN interval
DIN_MAX_STREAKS = 10           # Maximum streaks in the DIN interval

# Output resolution
SEGMENTATION_DPI = 600
DISTRIBUTION_DPI = 1000


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_closest_contour_point(contour, corner):
    """
    Find the point on a contour that is closest to a given corner.

    This is used to identify the four corner points of the sample region
    so we can apply a perspective correction (rectification).

    Args:
        contour: Contour array of shape (N, 1, 2) from cv2.findContours.
        corner:  Target corner as a numpy array [x, y].

    Returns:
        Tuple (x, y) of the closest contour point.
    """
    distances = np.sqrt(((contour - corner) ** 2).sum(axis=2))
    return tuple(contour[np.argmin(distances)][0])


def position_mm_to_particle_size_um(position_mm):
    """
    Convert a position along the grindometer groove (in mm from the deep end)
    to particle size in micrometers.

    At position 0 mm (deep end):   particle size = 50 µm
    At position 145 mm (shallow end): particle size = 0 µm

    The relationship is linear:
        particle_size = MAX_SIZE * (1 - position / GROOVE_LENGTH)

    Args:
        position_mm: Distance from the deep end in mm (scalar or array).

    Returns:
        Particle size in micrometers.
    """
    return MAX_PARTICLE_SIZE_UM * (1 - position_mm / GROOVE_LENGTH_MM)


def load_probe_image(data_folder, probe_number):
    """
    Load the image file for a given probe number from the data folder.

    Args:
        data_folder:  Path to the folder containing probe images.
        probe_number: String identifier for the probe (e.g. "5022").

    Returns:
        The loaded image as a BGR numpy array.

    Raises:
        FileNotFoundError: If no image matching the probe number is found.
    """
    image_files = os.listdir(data_folder)
    selected_file = next(
        (f for f in image_files if f"Probe{probe_number}" in f), None
    )
    if not selected_file:
        raise FileNotFoundError(
            f"No image file found for Probe {probe_number} in {data_folder}"
        )

    image_path = os.path.join(data_folder, selected_file)
    print(f"Analyzing image: {selected_file}")

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Failed to load image: {image_path}")
    return image


def detect_sample_contour(image):
    """
    Detect the largest contour that represents the grindometer sample region.

    Uses adaptive thresholding on the HSV saturation and value channels,
    then finds contours and filters by area to isolate the sample.

    Args:
        image: BGR input image.

    Returns:
        The largest valid contour as a numpy array.

    Raises:
        ValueError: If no suitable contour is found.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)

    # Adaptive thresholding highlights local contrast variations,
    # which effectively detects the edges of the sample region
    saturation_mask = cv2.adaptiveThreshold(
        saturation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C
    )
    value_mask = cv2.adaptiveThreshold(
        value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C
    )

    # Combine both masks — a pixel is foreground if either channel detects it
    combined_mask = value_mask + saturation_mask

    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    max_pixels = image.shape[0] * image.shape[1]
    valid_contours = [
        c for c in contours
        if MIN_CONTOUR_AREA < cv2.contourArea(c) < max_pixels * MAX_CONTOUR_FRACTION
    ]
    print(f"Found {len(valid_contours)} valid contours")

    if not valid_contours:
        raise ValueError("No valid sample contour found in the image.")

    return max(valid_contours, key=cv2.contourArea)


def rectify_image(image, contour):
    """
    Apply a perspective transform to straighten the sample region.

    The sample may be photographed at an angle. This function finds the
    four corner points of the sample contour (closest points to the image
    corners), then computes and applies a perspective warp so that the
    sample fills the entire frame, correcting for any tilt or skew.

    Args:
        image:   BGR input image.
        contour: The sample contour from detect_sample_contour().

    Returns:
        Rectified image resized to RECTIFIED_IMAGE_SIZE.
    """
    img_h, img_w = image.shape[:2]
    image_corners = {
        "top_left":     np.array([0, 0]),
        "top_right":    np.array([img_w - 1, 0]),
        "bottom_left":  np.array([0, img_h - 1]),
        "bottom_right": np.array([img_w - 1, img_h - 1]),
    }

    reshaped = contour.reshape(-1, 1, 2)
    source_points = np.float32([
        find_closest_contour_point(reshaped, image_corners["top_left"]),
        find_closest_contour_point(reshaped, image_corners["top_right"]),
        find_closest_contour_point(reshaped, image_corners["bottom_left"]),
        find_closest_contour_point(reshaped, image_corners["bottom_right"]),
    ])
    dest_points = np.float32([
        image_corners["top_left"],
        image_corners["top_right"],
        image_corners["bottom_left"],
        image_corners["bottom_right"],
    ])

    transform = cv2.getPerspectiveTransform(source_points, dest_points)
    rectified = cv2.warpPerspective(image, transform, (img_w, img_h))
    rectified = cv2.resize(rectified, RECTIFIED_IMAGE_SIZE)
    return rectified


def extract_roi(rectified_image):
    """
    Extract the Region of Interest (measurement groove) from the rectified image.

    The ROI boundaries are defined by the constants ROI_X_START, ROI_X_END,
    ROI_Y_START, ROI_Y_END. These correspond to the physical groove area
    in the standardized rectified image.

    Args:
        rectified_image: Perspective-corrected image at RECTIFIED_IMAGE_SIZE.

    Returns:
        Cropped BGR image of the measurement groove.
    """
    return rectified_image[ROI_Y_START:ROI_Y_END, ROI_X_START:ROI_X_END]


def segment_streaks_gmm(roi_image):
    """
    Segment streaks from background using a Gaussian Mixture Model.

    The V (value/brightness) channel of the HSV image is used because
    streaks appear as darker regions against a lighter background.

    A 2-component GMM separates the pixel intensities into two clusters:
    - The cluster with lower mean brightness = streaks (dark particles)
    - The cluster with higher mean brightness = background (light groove)

    Args:
        roi_image: BGR image of the measurement groove.

    Returns:
        Binary image (uint8) where 255 = streak, 0 = background.
    """
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]  # Brightness channel

    pixel_values = v_channel.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(pixel_values)

    labels = gmm.predict(pixel_values).reshape(v_channel.shape)

    # The cluster with the lower mean brightness is the streak cluster
    # (particles are darker than the groove background)
    streak_label = 0 if gmm.means_[0] < gmm.means_[1] else 1
    binary = np.where(labels == streak_label, 255, 0).astype(np.uint8)

    return binary


def count_streaks_per_slice(binary_image):
    """
    Count the number of valid streaks in each 1 mm vertical slice.

    The image is divided into horizontal slices (each SLICE_HEIGHT_MM tall).
    For each slice, connected components are found and filtered by minimum
    size (MIN_STREAK_WIDTH_PX) to exclude noise. A connected component
    counts as a streak if its width OR height exceeds the threshold.

    Args:
        binary_image: Binary image from segment_streaks_gmm().

    Returns:
        List of streak counts, one per slice (from top/deep end to bottom).
    """
    img_height, _ = binary_image.shape
    mm_per_pixel = GROOVE_LENGTH_MM / img_height
    slice_height_px = max(int(SLICE_HEIGHT_MM / mm_per_pixel), 1)
    num_slices = img_height // slice_height_px

    streak_counts = []
    for i in range(num_slices):
        y_start = i * slice_height_px
        y_end = min(y_start + slice_height_px, img_height)
        slice_img = binary_image[y_start:y_end, :]

        if slice_img.ndim != 2 or slice_img.shape[0] == 0 or slice_img.shape[1] == 0:
            streak_counts.append(0)
            continue

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(slice_img)
        count = 0
        for j in range(1, num_labels):  # Skip label 0 (background)
            comp_w = stats[j, cv2.CC_STAT_WIDTH]
            comp_h = stats[j, cv2.CC_STAT_HEIGHT]
            if comp_w >= MIN_STREAK_WIDTH_PX or comp_h >= MIN_STREAK_WIDTH_PX:
                count += 1
        streak_counts.append(count)

    return streak_counts


def find_max_particle_size(streak_counts):
    """
    Find the maximum particle size — the position of the first visible streak.

    The first streak (scanning from the deep end) indicates the largest
    particle in the sample: it is the first position where a particle is
    too large to fit in the groove.

    Args:
        streak_counts: List of streak counts per slice.

    Returns:
        Tuple (particle_size_um, slice_index) or (None, None) if no streaks found.
    """
    for idx, count in enumerate(streak_counts):
        if count > 0:
            position_mm = idx * SLICE_HEIGHT_MM
            size_um = position_mm_to_particle_size_um(position_mm)
            print(f"First streak at position {position_mm:.1f} mm "
                  f"→ max particle size: {size_um:.1f} µm")
            return size_um, idx

    print("No valid streaks found in the image.")
    return None, None


def find_mean_particle_size_din(streak_counts, start_idx):
    """
    Find the mean particle size according to DIN standard.

    The DIN method defines the mean particle size ("Mahlfeinheit") as the
    position of the first 3 mm interval (starting from the max particle
    position) that contains between 5 and 10 streaks.

    Args:
        streak_counts: List of streak counts per slice.
        start_idx:     Slice index of the first streak (from find_max_particle_size).

    Returns:
        Particle size in µm, or None if no valid interval is found.
    """
    interval_slices = int(DIN_INTERVAL_MM / SLICE_HEIGHT_MM)

    for idx in range(start_idx, len(streak_counts) - interval_slices + 1):
        total = sum(streak_counts[idx:idx + interval_slices])
        if DIN_MIN_STREAKS <= total <= DIN_MAX_STREAKS:
            position_mm = idx * SLICE_HEIGHT_MM
            size_um = position_mm_to_particle_size_um(position_mm)
            print(f"DIN mean fineness at position {position_mm:.1f} mm "
                  f"→ mean particle size: {size_um:.1f} µm")
            return size_um

    print("No 3 mm interval found with 5–10 streaks (DIN criterion not met).")
    return None


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_segmentation(binary_image, probe_number):
    """
    Display and save the GMM segmentation result with a micrometer scale.

    The Y-axis is labeled with particle sizes in µm (50 at top, 0 at bottom),
    mapping the physical groove depth to pixel positions.
    """
    img_height = binary_image.shape[0]

    plt.figure(figsize=(8, 8))
    plt.imshow(binary_image, cmap="gray")
    plt.title(f"GMM Segmentation — Probe {probe_number}")

    # Create micrometer scale on the right Y-axis
    tick_positions = np.linspace(0, img_height, num=6)
    tick_mm = tick_positions * (GROOVE_LENGTH_MM / img_height)
    tick_um = position_mm_to_particle_size_um(tick_mm)

    ax = plt.gca()
    ax.set_ylim([img_height, 0])  # Invert Y so deep end (50 µm) is at top
    ax.yaxis.set_ticks(tick_positions)
    ax.yaxis.set_ticklabels([f"{int(s)} µm" for s in tick_um])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_label_text("")

    filename = f"{OUTPUT_FOLDER}GMM_Segmentierung_Probe{probe_number}.png"
    plt.savefig(filename, dpi=SEGMENTATION_DPI, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")


def plot_streak_distribution(streak_counts, max_size_um, mean_size_um, probe_number):
    """
    Plot the number of streaks vs. particle size and annotate key metrics.

    The X-axis shows particle size in µm (50 on the left, 0 on the right),
    and the Y-axis shows the number of streaks detected per 1 mm slice.
    Vertical lines mark the maximum particle size (red) and mean fineness (green).
    """
    num_slices = len(streak_counts)
    positions_mm = np.array([i * SLICE_HEIGHT_MM for i in range(num_slices)])
    particle_sizes = position_mm_to_particle_size_um(positions_mm)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.figure(figsize=(10, 6))

    sns.lineplot(x=particle_sizes, y=streak_counts,
                 marker="o", color="navy", linewidth=2.5)

    plt.xlabel("Particle Size (µm)")
    plt.ylabel("Number of Streaks")
    plt.gca().invert_xaxis()  # 50 µm on the left, 0 µm on the right
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(np.arange(0, 55, 5))
    plt.xlim(50, 0)

    peak_count = max(streak_counts) if streak_counts else 1

    # Annotate maximum particle size (first streak position)
    if max_size_um is not None:
        plt.axvline(x=max_size_um, color="red", linestyle="--",
                    label="Max Particle Size")
        plt.text(max_size_um + 1, peak_count * 0.8,
                 f"{max_size_um:.1f} µm",
                 color="red", rotation=90, va="center")

    # Annotate DIN mean fineness
    if mean_size_um is not None:
        plt.axvline(x=mean_size_um, color="green", linestyle="-.",
                    label="Mean Fineness (DIN)")
        plt.text(mean_size_um + 1, peak_count * 0.6,
                 f"{mean_size_um:.1f} µm",
                 color="green", rotation=90, va="center")

    plt.legend()
    plt.tight_layout()

    filename = f"{OUTPUT_FOLDER}Streifenverteilung_Probe{probe_number}.png"
    plt.savefig(filename, dpi=DISTRIBUTION_DPI, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the full particle size measurement pipeline."""

    # Ensure output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Step 1: Load the probe image
    image = load_probe_image(DATA_FOLDER, PROBE_NUMBER)

    # Step 2: Detect the sample region and find its boundary contour
    contour = detect_sample_contour(image)

    # Step 3: Apply perspective correction to straighten the sample
    rectified = rectify_image(image, contour)

    # Step 4: Extract the measurement groove (ROI)
    roi = extract_roi(rectified)

    # Step 5: Segment streaks using Gaussian Mixture Model
    binary = segment_streaks_gmm(roi)

    # Step 6: Save and display the segmentation result
    plot_segmentation(binary, PROBE_NUMBER)

    # Step 7: Count streaks in each 1 mm slice
    streak_counts = count_streaks_per_slice(binary)

    # Step 8: Determine maximum particle size (first streak)
    max_size_um, max_idx = find_max_particle_size(streak_counts)

    # Step 9: Determine mean particle size per DIN standard
    mean_size_um = None
    if max_idx is not None:
        mean_size_um = find_mean_particle_size_din(streak_counts, max_idx)

    # Step 10: Plot and save the streak distribution
    plot_streak_distribution(streak_counts, max_size_um, mean_size_um, PROBE_NUMBER)

    # Final summary
    print("\n=== Results ===")
    if max_size_um is not None:
        print(f"  Maximum particle size:        {max_size_um:.1f} µm")
    if mean_size_um is not None:
        print(f"  Mean particle size (DIN):      {mean_size_um:.1f} µm")


if __name__ == "__main__":
    main()
