############################################################
# GFP-only condensate detection in overexpression cells
#
# Uses ONLY channel 0 = GFP
# Uses ONLY StarDist for nuclear/cell segmentation
# Condensates are detected inside StarDist masks
#
# Condensate detection uses hysteresis:
#   - high threshold = confident bright seed
#   - low threshold  = expanded/dimmer condensate body
#
# Quantification is always done on original GFP values.
############################################################

############################
# USER SETTINGS
############################

SAVE_EXCEL = True
SAVE_PLOTS_PDF = True
SKIP_FILES_ALREADY_ANALYZED = False
REMOVE_FINAL_NUMBER_FOR_GROUPING = True

RUN_STATS = True
CONTROL_MATCH_TEXT = "DMSO"
STATS_TEST = "mannwhitney"  # "mannwhitney" or "ttest"
STATS_LEVEL = ("file"   )        # "file" or "cell"

# Plot points level:
# "file" = each point is one ND2 file/replicate
# "cell" = each point is one segmented cell/nucleus
PLOT_POINTS_LEVEL = "file"

############################
# CHANNEL SETTINGS
############################

GFP_CHANNEL = 0

############################
# STARDIST PREPROCESSING
############################

SEG_CLIP_HIGH_PERCENTILE = 99
SEG_SMOOTH_SIGMA = 6

STARDIST_NORM_LOW = 1
STARDIST_NORM_HIGH = 99.8

############################
# NUCLEUS / CELL FILTERING
############################

MIN_NUC_AREA = 600
MAX_NUC_AREA = 30000
NUC_INT_PCT = 0
NUC_GFP_MIN_MEAN = 0

############################
# CONDENSATE DETECTION
############################

USE_MAX_FILTER_FOR_DETECTION = True
MAX_FILTER_SIZE = 3

COND_LOW_INT_FACTOR = 0.6
COND_LOW_PERCENTILE = 90

COND_HIGH_INT_FACTOR = 1.5
COND_HIGH_PERCENTILE = 90

COND_MIN_AREA = 30
COND_MAX_AREA = 12000
COND_MEAN_OVER_NUC_MEAN = 1
COND_MAX_OVER_NUC_MEAN = 1

MIN_COND_INTEGRATED_DENSITY_OVER_NUC_MEAN = 80

USE_CIRCULARITY_FILTER = True
COND_MIN_CIRC = 0.50

############################
# IMPORTS
############################

import os
import re
import time
import warnings
import logging

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tkinter import Tk, filedialog
from scipy.ndimage import maximum_filter, gaussian_filter, binary_fill_holes
from scipy.stats import mannwhitneyu, ttest_ind

from skimage import exposure, measure, morphology
from skimage.segmentation import clear_border
from skimage.morphology import disk

from stardist.models import StarDist2D
from csbdeep.utils import normalize
from nd2reader import ND2Reader

############################
# CLEANUP
############################

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

############################
# SELECT FOLDER
############################

def select_folder():
    root = Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    root.focus_force()

    folder = filedialog.askdirectory(
        parent=root,
        title="Select folder containing ND2 files"
    )

    root.destroy()

    if folder == "":
        raise ValueError("No folder selected.")

    return folder


print("\nSelect folder containing ND2 files...\n")
input_folder = select_folder()
print(f"Selected folder:\n{input_folder}\n")

############################
# OUTPUT FOLDERS
############################

analysis_folder = os.path.join(input_folder, "Analysis")
overlay_folder = os.path.join(analysis_folder, "Overlays")
plot_folder = os.path.join(analysis_folder, "Plots")

os.makedirs(analysis_folder, exist_ok=True)
os.makedirs(overlay_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

############################
# FIND FILES
############################

nd2_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(".nd2")
])

if len(nd2_files) == 0:
    raise ValueError("No ND2 files found.")

print(f"Found {len(nd2_files)} ND2 files.\n")

############################
# LOAD STARDIST
############################

print("Loading StarDist model...")
model = StarDist2D.from_pretrained("2D_versatile_fluo")
print("StarDist loaded.\n")

summary_records = []

############################
# HELPER FUNCTIONS
############################

def get_group_name(filename):
    base = os.path.splitext(filename)[0]

    if REMOVE_FINAL_NUMBER_FOR_GROUPING:
        base = re.sub(r"([_-])\d+$", "", base)

    return base


def p_to_stars(p):
    if pd.isna(p):
        return "NA"
    if p <= 0.0001:
        return "****"
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "ns"


def calculate_circularity(area, perimeter):
    if perimeter == 0:
        return 0

    return 4 * np.pi * area / (perimeter ** 2)


def load_nd2_channel_max_projection(path, channel_index):
    with ND2Reader(path) as reader:

        n_z = reader.sizes.get("z", 1)
        n_t = reader.sizes.get("t", 1)

        kwargs = {}

        if "c" in reader.sizes:
            kwargs["c"] = channel_index
        if "z" in reader.sizes:
            kwargs["z"] = 0
        if "t" in reader.sizes:
            kwargs["t"] = 0

        probe = reader.get_frame_2D(**kwargs).astype(np.float32)
        max_proj = np.zeros_like(probe, dtype=np.float32)

        for t in range(n_t):
            for z in range(n_z):

                frame_kwargs = {}

                if "c" in reader.sizes:
                    frame_kwargs["c"] = channel_index
                if "z" in reader.sizes:
                    frame_kwargs["z"] = z
                if "t" in reader.sizes:
                    frame_kwargs["t"] = t

                frame = reader.get_frame_2D(**frame_kwargs).astype(np.float32)
                np.maximum(max_proj, frame, out=max_proj)

    return max_proj


def segment_with_stardist(gfp):
    clipped = np.clip(
        gfp,
        0,
        np.percentile(gfp, SEG_CLIP_HIGH_PERCENTILE)
    )

    smooth = gaussian_filter(
        clipped,
        sigma=SEG_SMOOTH_SIGMA
    )

    norm_img = normalize(
        smooth,
        STARDIST_NORM_LOW,
        STARDIST_NORM_HIGH,
        axis=(0, 1)
    )

    labels, _ = model.predict_instances(norm_img)

    labels = clear_border(labels)

    labels = morphology.remove_small_objects(
        labels,
        min_size=MIN_NUC_AREA
    )

    props = measure.regionprops(labels)

    clean = np.zeros_like(labels, dtype=np.int32)

    new_label = 1

    for p in props:

        if p.area < MIN_NUC_AREA:
            continue

        if p.area > MAX_NUC_AREA:
            continue

        clean[labels == p.label] = new_label
        new_label += 1

    return clean


def get_hysteresis_thresholds(nuclear_pixels):
    low_mean_thr = (
        nuclear_pixels.mean()
        + nuclear_pixels.std() * COND_LOW_INT_FACTOR
    )

    low_pct_thr = np.percentile(
        nuclear_pixels,
        COND_LOW_PERCENTILE
    )

    high_mean_thr = (
        nuclear_pixels.mean()
        + nuclear_pixels.std() * COND_HIGH_INT_FACTOR
    )

    high_pct_thr = np.percentile(
        nuclear_pixels,
        COND_HIGH_PERCENTILE
    )

    low_threshold = max(low_mean_thr, low_pct_thr)
    high_threshold = max(high_mean_thr, high_pct_thr)

    if low_threshold >= high_threshold:
        low_threshold = high_threshold * 0.75

    return low_threshold, high_threshold


def make_hysteresis_condensate_mask(roi_detection, roi_mask):
    nuclear_detection_pixels = roi_detection[roi_mask]

    low_threshold, high_threshold = get_hysteresis_thresholds(
        nuclear_detection_pixels
    )

    low_mask = roi_detection >= low_threshold
    high_mask = roi_detection >= high_threshold

    low_mask[~roi_mask] = False
    high_mask[~roi_mask] = False

    low_mask = morphology.remove_small_objects(
        low_mask,
        min_size=COND_MIN_AREA
    )

    low_mask = binary_fill_holes(low_mask)

    low_labels = measure.label(low_mask)

    final_mask = np.zeros_like(low_mask, dtype=bool)

    props = measure.regionprops(low_labels)

    for p in props:

        component_mask = low_labels == p.label

        has_high_seed = np.any(high_mask[component_mask])

        if not has_high_seed:
            continue

        final_mask[component_mask] = True

    final_mask = morphology.remove_small_objects(
        final_mask,
        min_size=COND_MIN_AREA
    )

    return final_mask


def get_stats_input_table(summary_df, per_replicate_summary):
    if STATS_LEVEL == "cell":
        return summary_df.copy()

    if STATS_LEVEL == "file":
        stats_df = per_replicate_summary.copy()

        stats_df = stats_df.rename(
            columns={
                "Mean_Condensates": "Num_Condensates",
                "Mean_CF_Area": "Condensed_Fraction_Area",
                "Mean_CF_Intensity": "Condensed_Fraction_Intensity"
            }
        )

        return stats_df

    raise ValueError("STATS_LEVEL must be 'file' or 'cell'")


def get_plot_input_table(summary_df, per_replicate_summary, value_col):
    if PLOT_POINTS_LEVEL == "cell":
        plot_df = summary_df[["Group", "File", value_col]].copy()
        plot_df = plot_df.rename(columns={value_col: "Value"})
        return plot_df

    if PLOT_POINTS_LEVEL == "file":
        file_df = per_replicate_summary.rename(
            columns={
                "Mean_Condensates": "Num_Condensates",
                "Mean_CF_Area": "Condensed_Fraction_Area",
                "Mean_CF_Intensity": "Condensed_Fraction_Intensity"
            }
        )

        plot_df = file_df[["Group", "File", value_col]].copy()
        plot_df = plot_df.rename(columns={value_col: "Value"})
        return plot_df

    raise ValueError("PLOT_POINTS_LEVEL must be 'file' or 'cell'")


def run_stats_against_control(df, metric):
    control_groups = [
        g for g in df["Group"].unique()
        if CONTROL_MATCH_TEXT.lower() in g.lower()
    ]

    if len(control_groups) == 0:
        raise ValueError(
            f"No control group found containing: {CONTROL_MATCH_TEXT}"
        )

    if len(control_groups) > 1:
        print(
            f"Warning: multiple control groups found: {control_groups}. "
            f"Using first one: {control_groups[0]}"
        )

    control_group = control_groups[0]

    control_values = df.loc[
        df["Group"] == control_group,
        metric
    ].dropna()

    results = []

    for group in sorted(df["Group"].unique()):

        test_values = df.loc[
            df["Group"] == group,
            metric
        ].dropna()

        if group == control_group:
            p_value = np.nan
            statistic = np.nan
            stars = "Control"

        elif len(control_values) < 2 or len(test_values) < 2:
            p_value = np.nan
            statistic = np.nan
            stars = "NA"

        else:
            if STATS_TEST == "mannwhitney":
                stat_result = mannwhitneyu(
                    control_values,
                    test_values,
                    alternative="two-sided"
                )

                statistic = stat_result.statistic
                p_value = stat_result.pvalue

            elif STATS_TEST == "ttest":
                stat_result = ttest_ind(
                    control_values,
                    test_values,
                    equal_var=False
                )

                statistic = stat_result.statistic
                p_value = stat_result.pvalue

            else:
                raise ValueError("STATS_TEST must be 'mannwhitney' or 'ttest'")

            stars = p_to_stars(p_value)

        results.append({
            "Stats_Level": STATS_LEVEL,
            "Metric": metric,
            "Control_Group": control_group,
            "Compared_Group": group,
            "Test": STATS_TEST,
            "N_Control": len(control_values),
            "N_Group": len(test_values),
            "Control_Mean": control_values.mean(),
            "Group_Mean": test_values.mean(),
            "Statistic": statistic,
            "P_Value": p_value,
            "Stars": stars
        })

    return pd.DataFrame(results)

############################
# PROCESS FILE
############################

def process_file(fname, idx, total):

    print(f"[{idx}/{total}] Processing {fname}")

    start = time.perf_counter()

    path = os.path.join(input_folder, fname)
    base_name = os.path.splitext(fname)[0]
    group_name = get_group_name(fname)

    overlay_path = os.path.join(
        overlay_folder,
        f"{base_name}_overlay.tif"
    )

    if SKIP_FILES_ALREADY_ANALYZED and os.path.exists(overlay_path):
        print("  Skipping existing overlay.\n")
        return

    gfp = load_nd2_channel_max_projection(
        path,
        GFP_CHANNEL
    )

    gfp8 = exposure.rescale_intensity(
        gfp,
        in_range="image",
        out_range=(0, 255)
    ).astype(np.uint8)

    labels = segment_with_stardist(gfp)

    nuc_props_all = measure.regionprops(
        labels,
        intensity_image=gfp
    )

    if len(nuc_props_all) == 0:
        print("  No StarDist objects detected.\n")
        return

    gfp_means = np.array([
        p.mean_intensity
        for p in nuc_props_all
    ])

    cutoff = np.percentile(
        gfp_means,
        NUC_INT_PCT
    )

    nuclei = [
        p for p in nuc_props_all
        if p.mean_intensity >= cutoff
        and p.mean_intensity >= NUC_GFP_MIN_MEAN
    ]

    print(f"  StarDist objects kept: {len(nuclei)}")

    overlay = np.zeros(
        (*gfp8.shape, 3),
        dtype=np.uint8
    )

    overlay[..., 1] = gfp8

    total_condensates = 0
    per_file_records = []

    for p in nuclei:

        nucleus_mask = labels == p.label

        nuclear_pixels = gfp[nucleus_mask]

        nuclear_gfp_mean = float(np.mean(nuclear_pixels))
        nuclear_gfp_median = float(np.median(nuclear_pixels))
        nuclear_gfp_intden = float(np.sum(nuclear_pixels))
        nuclear_area = int(np.sum(nucleus_mask))

        minr, minc, maxr, maxc = p.bbox

        roi_gfp = gfp[minr:maxr, minc:maxc]
        roi_mask = p.image

        if USE_MAX_FILTER_FOR_DETECTION:
            roi_detection = maximum_filter(
                roi_gfp,
                size=MAX_FILTER_SIZE
            )
        else:
            roi_detection = roi_gfp.copy()

        bw = make_hysteresis_condensate_mask(
            roi_detection,
            roi_mask
        )

        lbl = measure.label(bw)

        cond_props = measure.regionprops(
            lbl,
            intensity_image=roi_gfp
        )

        num_cond = 0
        sum_cond_area = 0
        sum_cond_intden = 0
        sum_cond_mean = 0
        sum_cond_circ = 0
        sum_cond_feret_ratio = 0

        for c in cond_props:

            area = c.area

            if area < COND_MIN_AREA:
                continue

            if area > COND_MAX_AREA:
                continue

            perim = c.perimeter or 1
            circ = calculate_circularity(area, perim)

            if USE_CIRCULARITY_FILTER and circ < COND_MIN_CIRC:
                continue

            cond_pixels = roi_gfp[lbl == c.label]

            cond_mean = float(np.mean(cond_pixels))
            cond_max = float(np.max(cond_pixels))
            cond_intden = float(np.sum(cond_pixels))

            if cond_mean < nuclear_gfp_mean * COND_MEAN_OVER_NUC_MEAN:
                continue

            if cond_max < nuclear_gfp_mean * COND_MAX_OVER_NUC_MEAN:
                continue

            if cond_intden < nuclear_gfp_mean * MIN_COND_INTEGRATED_DENSITY_OVER_NUC_MEAN:
                continue

            feret_max = getattr(c, "feret_diameter_max", None)

            if feret_max is None or feret_max == 0:
                feret_max = float(getattr(c, "major_axis_length", 0.0)) or 1.0

            feret_min = float(getattr(c, "minor_axis_length", feret_max))
            feret_ratio = feret_min / feret_max if feret_max > 0 else 0.0

            num_cond += 1
            total_condensates += 1

            sum_cond_area += float(area)
            sum_cond_intden += cond_intden
            sum_cond_mean += cond_mean
            sum_cond_circ += float(circ)
            sum_cond_feret_ratio += float(feret_ratio)

            mask_c = (lbl == c.label).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask_c,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                cnt_shift = cnt + np.array(
                    [[[minc, minr]]],
                    dtype=np.int32
                )

                cv2.drawContours(
                    overlay,
                    [cnt_shift],
                    -1,
                    (255, 0, 255),
                    1
                )

        frac_area = (
            sum_cond_area / nuclear_area
            if nuclear_area > 0 else 0
        )

        frac_intensity = (
            sum_cond_intden / nuclear_gfp_intden
            if nuclear_gfp_intden > 0 else 0
        )

        mean_cond_mean = (
            sum_cond_mean / num_cond
            if num_cond > 0 else 0
        )

        mean_circ = (
            sum_cond_circ / num_cond
            if num_cond > 0 else 0
        )

        mean_feret_ratio = (
            sum_cond_feret_ratio / num_cond
            if num_cond > 0 else 0
        )

        per_file_records.append({
            "File": fname,
            "Group": group_name,
            "Cell_ID": int(p.label),
            "Nucleus_Area": float(nuclear_area),
            "Nucleus_GFP_Mean": float(nuclear_gfp_mean),
            "Nucleus_GFP_Median": float(nuclear_gfp_median),
            "Nucleus_GFP_Integrated_Density": float(nuclear_gfp_intden),
            "Num_Condensates": int(num_cond),
            "Sum_Condensate_Area": float(sum_cond_area),
            "Sum_Condensate_Integrated_Density": float(sum_cond_intden),
            "Mean_Condensate_Intensity": float(mean_cond_mean),
            "Mean_Condensate_Circularity": float(mean_circ),
            "Mean_Condensate_Feret_Ratio": float(mean_feret_ratio),
            "Condensed_Fraction_Area": float(frac_area),
            "Condensed_Fraction_Intensity": float(frac_intensity)
        })

        border = (
            morphology.binary_dilation(
                nucleus_mask,
                disk(1)
            )
            ^ nucleus_mask
        )

        overlay[border] = [255, 255, 255]

    summary_records.extend(per_file_records)

    overlay_bgr = cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        overlay_path,
        overlay_bgr
    )

    elapsed = time.perf_counter() - start

    print(f"  Condensates detected: {total_condensates}")
    print(f"  Done in {elapsed:.2f}s\n")

############################
# RUN ALL FILES
############################

for i, fname in enumerate(nd2_files, start=1):
    process_file(fname, i, len(nd2_files))

############################
# SAVE TABLES
############################

summary_df = pd.DataFrame(summary_records)

if len(summary_df) == 0:
    raise ValueError("No StarDist objects were quantified.")

per_cell_path = os.path.join(
    analysis_folder,
    "gfp_condensates_per_cell_summary.csv"
)

summary_df.to_csv(
    per_cell_path,
    index=False
)

per_replicate_summary = (
    summary_df
    .groupby(["Group", "File"], as_index=False)
    .agg(
        N_Cells=("Cell_ID", "count"),
        Mean_Condensates=("Num_Condensates", "mean"),
        SEM_Condensates=("Num_Condensates", "sem"),
        Mean_CF_Area=("Condensed_Fraction_Area", "mean"),
        SEM_CF_Area=("Condensed_Fraction_Area", "sem"),
        Mean_CF_Intensity=("Condensed_Fraction_Intensity", "mean"),
        SEM_CF_Intensity=("Condensed_Fraction_Intensity", "sem"),
        Mean_Nucleus_GFP=("Nucleus_GFP_Mean", "mean"),
        SEM_Nucleus_GFP=("Nucleus_GFP_Mean", "sem")
    )
)

per_replicate_path = os.path.join(
    analysis_folder,
    "gfp_condensates_per_replicate_summary.csv"
)

per_replicate_summary.to_csv(
    per_replicate_path,
    index=False
)

group_summary = (
    per_replicate_summary
    .groupby("Group", as_index=False)
    .agg(
        N_Files=("File", "nunique"),
        Total_Cells=("N_Cells", "sum"),
        Mean_Condensates=("Mean_Condensates", "mean"),
        SEM_Condensates=("Mean_Condensates", "sem"),
        Mean_CF_Area=("Mean_CF_Area", "mean"),
        SEM_CF_Area=("Mean_CF_Area", "sem"),
        Mean_CF_Intensity=("Mean_CF_Intensity", "mean"),
        SEM_CF_Intensity=("Mean_CF_Intensity", "sem"),
        Mean_Nucleus_GFP=("Mean_Nucleus_GFP", "mean"),
        SEM_Nucleus_GFP=("Mean_Nucleus_GFP", "sem")
    )
)

group_path = os.path.join(
    analysis_folder,
    "gfp_condensates_group_summary.csv"
)

group_summary.to_csv(
    group_path,
    index=False
)

############################
# STATISTICS
############################

stats_df = pd.DataFrame()
stats_input_df = pd.DataFrame()

if RUN_STATS:

    stats_input_df = get_stats_input_table(
        summary_df,
        per_replicate_summary
    )

    stats_metrics = [
        "Num_Condensates",
        "Condensed_Fraction_Area",
        "Condensed_Fraction_Intensity"
    ]

    stats_tables = []

    for metric in stats_metrics:
        stats_tables.append(
            run_stats_against_control(
                stats_input_df,
                metric
            )
        )

    stats_df = pd.concat(
        stats_tables,
        ignore_index=True
    )

    print("\n==============================")
    print("STATISTICS AGAINST CONTROL")
    print("==============================")
    print(f"Control match text: {CONTROL_MATCH_TEXT}")
    print(f"Test used: {STATS_TEST}")
    print(f"Stats level: {STATS_LEVEL}\n")
    print(stats_df.to_string(index=False))

############################
# EXCEL OUTPUT
############################

excel_path = os.path.join(
    analysis_folder,
    "gfp_condensate_analysis_with_statistics.xlsx"
)

if SAVE_EXCEL:

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Per_Cell",
            index=False
        )

        per_replicate_summary.to_excel(
            writer,
            sheet_name="Per_Replicate",
            index=False
        )

        group_summary.to_excel(
            writer,
            sheet_name="Grouped",
            index=False
        )

        if RUN_STATS:
            stats_input_df.to_excel(
                writer,
                sheet_name="Stats_Input",
                index=False
            )

            stats_df.to_excel(
                writer,
                sheet_name="Stats_vs_Control",
                index=False
            )

    print(f"\nExcel file saved:\n{excel_path}")

############################
# PLOTS
############################

def get_stars_for_metric(metric, group):
    if not RUN_STATS or len(stats_df) == 0:
        return ""

    match = stats_df.loc[
        (stats_df["Metric"] == metric)
        & (stats_df["Compared_Group"] == group),
        "Stars"
    ]

    if len(match) == 0:
        return ""

    stars = str(match.iloc[0])

    if stars in ["Control", "NA"]:
        return ""

    return stars


def grouped_plot(value_col, ylabel, filename):

    plot_df = get_plot_input_table(
        summary_df,
        per_replicate_summary,
        value_col
    )

    group_means = (
        plot_df
        .groupby("Group", as_index=False)
        .agg(
            Mean=("Value", "mean"),
            SEM=("Value", "sem")
        )
    )

    groups = list(group_means["Group"])
    x = np.arange(len(groups))

    fig, ax = plt.subplots(
        figsize=(max(6, len(groups) * 1.2), 5)
    )

    ax.bar(
        x,
        group_means["Mean"],
        yerr=group_means["SEM"],
        capsize=5,
        edgecolor="black"
    )

    ymax = 0

    for i, group in enumerate(groups):

        vals = plot_df.loc[
            plot_df["Group"] == group,
            "Value"
        ].values

        vals = vals[~pd.isna(vals)]

        if len(vals) > 0:
            ymax = max(ymax, np.nanmax(vals))

        jitter = np.random.normal(
            0,
            0.05,
            len(vals)
        )

        ax.scatter(
            np.repeat(i, len(vals)) + jitter,
            vals,
            s=40,
            edgecolor="black",
            zorder=3
        )

    y_range = max(
        ymax,
        group_means["Mean"].max()
    ) * 0.15

    if y_range == 0:
        y_range = 1

    for i, group in enumerate(groups):

        stars = get_stars_for_metric(
            value_col,
            group
        )

        if stars == "":
            continue

        mean_value = group_means.loc[
            group_means["Group"] == group,
            "Mean"
        ].iloc[0]

        sem_value = group_means.loc[
            group_means["Group"] == group,
            "SEM"
        ].iloc[0]

        if pd.isna(sem_value):
            sem_value = 0

        y_pos = mean_value + sem_value + y_range * 0.25

        ax.text(
            i,
            y_pos,
            stars,
            ha="center",
            va="bottom",
            fontsize=14
        )

    ax.set_ylim(
        0,
        max(
            ymax,
            group_means["Mean"].max()
        ) + y_range
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        groups,
        rotation=45,
        ha="right"
    )

    ax.set_ylabel(ylabel)

    ax.set_title(
        f"Points shown by: {PLOT_POINTS_LEVEL}"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    outpath = os.path.join(
        plot_folder,
        filename
    )

    plt.savefig(outpath)
    plt.close()

    print(f"Plot saved:\n{outpath}")


if SAVE_PLOTS_PDF:

    grouped_plot(
        "Condensed_Fraction_Intensity",
        "Condensed fraction intensity",
        "condensed_fraction_intensity.pdf"
    )

    grouped_plot(
        "Condensed_Fraction_Area",
        "Condensed fraction area",
        "condensed_fraction_area.pdf"
    )

    grouped_plot(
        "Num_Condensates",
        "Condensates per cell",
        "condensates_per_cell.pdf"
    )

############################
# FINAL
############################

print("\n✅ All done!")
print(f"\nPer-cell CSV:\n{per_cell_path}")
print(f"\nPer-replicate CSV:\n{per_replicate_path}")
print(f"\nGrouped CSV:\n{group_path}")

if SAVE_EXCEL:
    print(f"\nExcel with statistics:\n{excel_path}")

print(f"\nPlot points level: {PLOT_POINTS_LEVEL}")
print(f"\nStats level: {STATS_LEVEL}")
print(f"\nResults saved in:\n{analysis_folder}")