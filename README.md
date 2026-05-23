# GFP Condensate Quantification Pipeline

Python pipeline for automatic detection and quantification of GFP-positive condensates in overexpression microscopy datasets.

Designed for:
- ND2 microscopy files
- GFP-only datasets
- Large bright condensates
- Nuclear/cell segmentation using StarDist
- Per-cell and per-file quantification
- Statistical comparison between conditions

Originally optimized for biomolecular condensate analysis in overexpression systems.


------------------------------------------------------------
FEATURES
------------------------------------------------------------

✓ Automatic folder selection

✓ Automatic creation of:
    Analysis/
    Analysis/Overlays/
    Analysis/Plots/

✓ Reads ND2 files directly

✓ Maximum projection of Z/T stacks

✓ StarDist segmentation of GFP-positive nuclei/cells

✓ Hysteresis-based condensate detection:
    - detects both bright and dim condensate regions
    - avoids fragmented condensate cores

✓ Filters fake condensates using:
    - area
    - circularity
    - integrated intensity
    - enrichment over nuclear background

✓ Quantifies:
    - number of condensates
    - condensate area
    - condensate integrated intensity
    - condensed fraction by area
    - condensed fraction by intensity

✓ Per-cell quantification

✓ Per-file replicate summaries

✓ Grouped condition summaries

✓ Statistics against control conditions

✓ Automatic significance stars

✓ PDF plots

✓ Excel export with multiple sheets


------------------------------------------------------------
INSTALLATION
------------------------------------------------------------

Recommended: create a clean conda environment.

Example:

conda create -n condensates python=3.10
conda activate condensates

Install packages:

pip install numpy pandas matplotlib scipy scikit-image opencv-python openpyxl tifffile nd2reader tensorflow stardist csbdeep


------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------

Run the script:

python your_script_name.py

A folder explorer will open.

Select the folder containing ND2 files.

The script will automatically create:

Analysis/

inside the selected folder.


------------------------------------------------------------
INPUT FILES
------------------------------------------------------------

Supported:
- .nd2 files

Current workflow assumes:
- Channel 0 = GFP

This can be changed in:

GFP_CHANNEL = 0


------------------------------------------------------------
OUTPUT FILES
------------------------------------------------------------

Analysis/

├── Overlays/
│   └── segmentation overlays

├── Plots/
│   └── PDF plots

├── gfp_condensates_per_cell_summary.csv

├── gfp_condensates_per_replicate_summary.csv

├── gfp_condensates_group_summary.csv

└── gfp_condensate_analysis_with_statistics.xlsx


------------------------------------------------------------
MAIN METRICS
------------------------------------------------------------

1. Num_Condensates
--------------------------------

Number of detected condensates per cell.


2. Condensed_Fraction_Area
--------------------------------

Fraction of nuclear area occupied by condensates.

Calculated as:

total condensate area / total nuclear area


3. Condensed_Fraction_Intensity
--------------------------------

Fraction of total nuclear GFP signal localized inside condensates.

Calculated as:

total condensate integrated intensity
/
total nuclear integrated intensity

This is usually the most biologically meaningful metric for condensate partitioning.


------------------------------------------------------------
SEGMENTATION
------------------------------------------------------------

Nuclear/cell segmentation is performed using:

StarDist2D
Model:
2D_versatile_fluo

Segmentation uses a preprocessed GFP image:
- intensity clipping
- smoothing
- normalization

IMPORTANT:
These preprocessing steps affect ONLY segmentation.

Quantification is always performed on the ORIGINAL raw GFP image.


------------------------------------------------------------
CONDENSATE DETECTION
------------------------------------------------------------

Condensates are detected using hysteresis thresholding.

This works in two steps:

1. High threshold
   Detects confident bright condensate seeds.

2. Low threshold
   Expands detection to dimmer condensate regions.

This improves detection of:
- large condensates
- dim condensates
- heterogeneous condensates

while avoiding:
- random bright noise
- fragmented condensate cores


------------------------------------------------------------
IMPORTANT TOGGLES
------------------------------------------------------------

Grouping:

REMOVE_FINAL_NUMBER_FOR_GROUPING = True

Example:

SampleA_001.nd2
SampleA_002.nd2

becomes:

SampleA


Statistics:

RUN_STATS = True

CONTROL_MATCH_TEXT = "DMSO"

STATS_LEVEL = "file"

Options:
- "file"
- "cell"


Plot points:

PLOT_POINTS_LEVEL = "file"

Options:
- "file"
- "cell"


------------------------------------------------------------
RECOMMENDED ANALYSIS STRATEGY
------------------------------------------------------------

Usually recommended:

Statistics:
    STATS_LEVEL = "file"

Plots:
    PLOT_POINTS_LEVEL = "file"

Reason:
Biological replicates are files, not cells.

Using cells directly can artificially inflate statistical power.


------------------------------------------------------------
TUNING CONDENSATE DETECTION
------------------------------------------------------------

If condensates are missed:
- lower thresholds
- lower percentile
- lower enrichment factors

If fake condensates appear:
- increase minimum area
- increase integrated intensity filter
- increase circularity filter


Most important settings:

COND_MIN_AREA
COND_MAX_AREA

COND_LOW_INT_FACTOR
COND_HIGH_INT_FACTOR

COND_LOW_PERCENTILE
COND_HIGH_PERCENTILE

MIN_COND_INTEGRATED_DENSITY_OVER_NUC_MEAN


------------------------------------------------------------
OVERLAY COLORS
------------------------------------------------------------

Green:
    GFP signal

White:
    StarDist nuclear/cell segmentation

Magenta:
    detected condensates


------------------------------------------------------------
NOTES
------------------------------------------------------------

This pipeline was optimized for:
- overexpression systems
- biomolecular condensates
- very bright condensates
- heterogeneous condensate intensity

It may require parameter tuning for:
- diffuse proteins
- low SNR datasets
- small puncta
- non-nuclear condensates


------------------------------------------------------------
AUTHOR
------------------------------------------------------------

Cassio Fleming
UMC Utrecht

If you use or modify this pipeline, please cite appropriately and validate parameters for your own biological system.
