import os
import glob
import h5py
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
from skimage.transform import resize
from tqdm import tqdm  # progress bar
import re
import torch

# --- Import Cellpose libraries ---
from cellpose import models, plot


# Pattern for TIFF files: {region}_{fov}_{z}_Fluorescence_{wavelength}_nm_Ex.tiff
TIFF_PATTERN = re.compile(
    r"([^_]+)_(\d+)_(\d+)_Fluorescence_(\d+)_nm_Ex\.tiff?", re.IGNORECASE
)

##############################
# 1. Z-Max Projection Function (Adapted for TIFF)
##############################
def zmax_projection(input_folder):
    """
    Process all acquisition folders (containing timepoint subdirectories),
    and generate a z-max projection across all timepoints for each FOV.
    
    The resulting TIFF images are saved in a subfolder named 'zmax'.
    """
    # Create output folder for zmax projections
    zmax_folder = os.path.join(input_folder, "zmax")
    os.makedirs(zmax_folder, exist_ok=True)
    
    # Get timepoint directories
    timepoint_dirs = sorted([d for d in os.listdir(input_folder) 
                           if os.path.isdir(os.path.join(input_folder, d)) 
                           and d.isdigit()], key=int)
    
    if not timepoint_dirs:
        print("No timepoint directories found.")
        return
    
    # Get all unique FOVs from first timepoint
    first_tp_path = os.path.join(input_folder, timepoint_dirs[0])
    tiff_files = glob.glob(os.path.join(first_tp_path, "*.tif*"))
    
    # Group files by region_fov
    fov_groups = {}
    for file_path in tiff_files:
        match = TIFF_PATTERN.match(os.path.basename(file_path))
        if match:
            region, fov, z, wavelength = match.groups()
            key = f"{region}_{fov}"
            if key not in fov_groups:
                fov_groups[key] = []
            fov_groups[key].append((region, int(fov), wavelength))
    
    # Process each FOV
    for fov_key in tqdm(fov_groups.keys(), desc="Processing FOVs for z-max projection"):
        region, fov, wavelength = fov_groups[fov_key][0]
        print(f"\nProcessing {fov_key} for z-max projection...")
        
        try:
            zmax = None  # To store the running maximum projection
            
            for tp_idx, tp_dir in enumerate(timepoint_dirs):
                tp_path = os.path.join(input_folder, tp_dir)
                
                # Find files for this FOV at this timepoint
                pattern = f"{region}_{fov}_*_Fluorescence_{wavelength}_nm_Ex.tif*"
                matching_files = glob.glob(os.path.join(tp_path, pattern))
                
                if not matching_files:
                    continue
                
                # If multiple z-slices, use middle one (like original script uses ResolutionLevel 0)
                if len(matching_files) > 1:
                    z_files = []
                    for f in matching_files:
                        match = TIFF_PATTERN.match(os.path.basename(f))
                        if match:
                            z_files.append((int(match.group(3)), f))
                    z_files.sort()
                    file_to_use = z_files[len(z_files)//2][1]
                else:
                    file_to_use = matching_files[0]
                
                # Read data (equivalent to reading from HDF5 TimePoint/Channel 0/Data)
                data = tifffile.imread(file_to_use)
                print(f"TimePoint {tp_idx}: Data shape {data.shape}, dtype {data.dtype}")
                
                if zmax is None:
                    zmax = data.astype(np.uint16)
                else:
                    zmax = np.maximum(zmax, data)
            
            if zmax is not None:
                print(f"Final z-max projection shape: {zmax.shape}, dtype: {zmax.dtype}")
                
                # Save the z-max projection TIFF
                output_filename = f"{fov_key}_zmax.tif"
                output_path = os.path.join(zmax_folder, output_filename)
                tifffile.imwrite(output_path, zmax)
                print(f"Saved z-max projection for {fov_key} to {output_path}")
        
        except Exception as e:
            print(f"Error processing {fov_key}: {e}")

##############################
# 2. Cellpose Segmentation Function (Unchanged)
##############################
def run_cellpose_segmentation_cyto2(input_folder, channels=(0, 0), diameter=None,
                                    flow_threshold=0.4, cellprob_threshold=0.0, gpu=True):
    """
    Run cellpose segmentation (cyto2) on TIFF images located in input_folder.
    Saves both mask images and high-resolution overlay images in a subfolder called 'mask_cyto2'.
    """
    # Create folder to store cellpose outputs
    mask_folder = os.path.join(input_folder, "mask_cyto2")
    os.makedirs(mask_folder, exist_ok=True)

    # Load the Cellpose model (cyto2) - Updated for new API with fallbacks
    model = None
    gpu_working = gpu
    
    try:
        # Try GPU first if requested
        if gpu:
            print("GPU detected:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
            model = models.CellposeModel(model_type='cyto2', gpu=True)
            print("Loaded CellposeModel with GPU=True")
        else:
            model = models.CellposeModel(model_type='cyto2', gpu=False)
            print("Loaded CellposeModel with GPU=False")
    except Exception as e:
        print(f"Failed to load model with GPU={gpu}: {e}")
        try:
            # Fallback to CPU
            print("Falling back to CPU...")
            model = models.CellposeModel(model_type='cyto2', gpu=False)
            gpu_working = False
            print("Loaded CellposeModel with GPU=False (fallback)")
        except Exception as e2:
            print(f"Failed to load model with CPU fallback: {e2}")
            return

    # List all TIFF files in the folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif')]
    if not image_files:
        print("No TIFF files found in", input_folder)
        return

    for image_file in tqdm(image_files, desc="Segmenting TIFF images"):
        try:
            image_path = os.path.join(input_folder, image_file)
            img = tifffile.imread(image_path)
            
            # Ensure image is in correct format (2D grayscale)
            if len(img.shape) > 2:
                if img.shape[2] > 1:
                    # Convert to grayscale if multichannel
                    img = img[:, :, 0]
                else:
                    img = img.squeeze()
            
            print(f"Processing {image_file}, shape: {img.shape}, dtype: {img.dtype}")

            # Run Cellpose segmentation - Handle version differences
            try:
                # Try new Cellpose API (v4.x): returns 3 values
                result = model.eval(
                    img,
                    diameter=diameter,
                    channels=[channels[0], channels[1]],
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold
                )
                
                if len(result) == 3:
                    masks, flows, styles = result
                    diams = None
                elif len(result) == 4:
                    masks, flows, styles, diams = result
                else:
                    print(f"Unexpected number of return values: {len(result)}")
                    continue
                    
            except Exception as eval_error:
                print(f"Error during model evaluation for {image_file}: {eval_error}")
                
                # If GPU fails, try CPU fallback for this image
                if gpu_working:
                    print("Trying CPU fallback for this image...")
                    try:
                        cpu_model = models.CellposeModel(model_type='cyto2', gpu=False)
                        result = cpu_model.eval(
                            img,
                            diameter=diameter,
                            channels=[channels[0], channels[1]],
                            flow_threshold=flow_threshold,
                            cellprob_threshold=cellprob_threshold
                        )
                        
                        if len(result) == 3:
                            masks, flows, styles = result
                            diams = None
                        elif len(result) == 4:
                            masks, flows, styles, diams = result
                        else:
                            print(f"CPU fallback also failed for {image_file}")
                            continue
                            
                    except Exception as cpu_error:
                        print(f"CPU fallback also failed for {image_file}: {cpu_error}")
                        continue
                else:
                    continue

            # Adjust mask size if necessary
            if masks.shape != img.shape:
                print(f"Resizing mask from {masks.shape} to match input image dimensions {img.shape}")
                masks = resize(masks, img.shape, order=0, preserve_range=True,
                               anti_aliasing=False).astype(np.uint16)
            else:
                print("Mask dimensions match input image dimensions.")

            # Save mask and overlay image
            base_name = os.path.splitext(image_file)[0]
            mask_filename = f"{base_name}_mask_cyto2.tif"
            mask_path = os.path.join(mask_folder, mask_filename)
            tifffile.imwrite(mask_path, masks.astype(np.uint16))

            # QC: Save the overlay visualization
            try:
                fig = plt.figure(figsize=(12, 5))
                plot.show_segmentation(fig, img, masks, flows[0], channels=[channels[0], channels[1]])
                plt.tight_layout()
                overlay_filename = f"{base_name}_overlay_cyto2.png"
                overlay_path = os.path.join(mask_folder, overlay_filename)
                plt.savefig(overlay_path, dpi=300)
                plt.close(fig)
                
                print(f"Processed {image_file}")
                print(f"  Mask: {mask_path}")
                print(f"  Overlay: {overlay_path}")
                
                # QC: Print number of cells detected
                n_cells = len(np.unique(masks)) - 1  # Subtract background
                print(f"  Cells detected: {n_cells}")
                
            except Exception as plot_error:
                print(f"Failed to create overlay plot for {image_file}: {plot_error}")
                print(f"  Mask saved: {mask_path}")
                n_cells = len(np.unique(masks)) - 1
                print(f"  Cells detected: {n_cells}")
                
        except Exception as e:
            print(f"Failed to process {image_file}: {e}")
            continue

##############################
# 3. Intensity Extraction Functions (Adapted for TIFF)
##############################
def extract_mean_intensities(acquisition_folder, mask_file, fov_key, channel="488"):
    """
    For each time point in the acquisition folder,
    compute the mean pixel intensity for each cell region defined in the mask.
    """
    # Parse region and fov from fov_key
    parts = fov_key.split('_')
    region = parts[0]
    fov = parts[1]
    
    # Get timepoint directories
    timepoint_dirs = sorted([d for d in os.listdir(acquisition_folder) 
                           if os.path.isdir(os.path.join(acquisition_folder, d)) 
                           and d.isdigit()], key=int)
    
    print("Found time points:", [f"TimePoint {i}" for i in range(len(timepoint_dirs))])
    
    mask = tifffile.imread(mask_file)
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # ignore background
    print("Unique mask labels:", unique_labels)
    
    results = {int(label): [] for label in unique_labels}
    time_points = []
    
    for tp_idx, tp_dir in enumerate(tqdm(timepoint_dirs, desc="Extracting intensities per time point")):
        tp_path = os.path.join(acquisition_folder, tp_dir)
        
        # Find the TIFF file for this timepoint (like accessing HDF5 path)
        pattern = f"{region}_{fov}_*_Fluorescence_{channel}_nm_Ex.tif*"
        matching_files = glob.glob(os.path.join(tp_path, pattern))
        
        if not matching_files:
            print(f"No file found for TimePoint {tp_idx}")
            continue
        
        # Use middle z if multiple
        if len(matching_files) > 1:
            z_files = []
            for f in matching_files:
                match = TIFF_PATTERN.match(os.path.basename(f))
                if match:
                    z_files.append((int(match.group(3)), f))
            z_files.sort()
            file_to_use = z_files[len(z_files)//2][1]
        else:
            file_to_use = matching_files[0]
        
        # Read image data
        image_data = tifffile.imread(file_to_use)
        
        if image_data.shape != mask.shape:
            print(f"Shape mismatch at TimePoint {tp_idx}: image {image_data.shape} vs mask {mask.shape}")
            continue
        
        for label in results.keys():
            region = image_data[mask == label]
            mean_val = np.mean(region) if region.size > 0 else np.nan
            results[label].append(mean_val)
            
        time_val = int(tp_idx)
        time_points.append(time_val)
            
    return np.array(time_points), results

def compute_f_f0(time_points, results, baseline_times=[5, 6, 7, 8]):
    """
    Compute Fâ‚€ for each cell as the mean intensity over the baseline time points,
    then compute F/Fâ‚€ for every time point.
    """
    f_f0 = {}
    time_points = np.array(time_points)
    baseline_mask = np.isin(time_points, baseline_times)
    if not np.any(baseline_mask):
        raise ValueError("None of the specified baseline time points were found in the data.")
    
    for cell_id, intensities in results.items():
        intensities = np.array(intensities, dtype=float)
        baseline = np.mean(intensities[baseline_mask])
        if baseline == 0:
            f_f0[cell_id] = intensities / 1e-6
        else:
            f_f0[cell_id] = intensities / baseline
    return f_f0

def process_pair(acquisition_folder, mask_file, output_dir, fov_key, channel="488", baseline_times=[5,6,7,8]):
    """
    Process one pair of acquisition folder and mask file:
      - Extract mean intensity data.
      - Compute F/Fâ‚€ values.
      - Save a CSV with columns: Time_min, Cell_ID, Raw_Intensity, F0, F/Fâ‚€.
    """
    print("Processing:", fov_key)
    time_points, results = extract_mean_intensities(acquisition_folder, mask_file, fov_key, channel)
    base_name = fov_key
    
    # Compute F/Fâ‚€ and Fâ‚€ per cell.
    f_f0 = compute_f_f0(time_points, results, baseline_times)
    f0_dict = {}
    for cell_id, intensities in results.items():
        intensities = np.array(intensities, dtype=float)
        baseline_val = np.mean(intensities[np.isin(time_points, baseline_times)])
        f0_dict[cell_id] = baseline_val
        
    # Build CSV rows.
    rows = []
    for cell_id in sorted(results.keys()):
        for i, t in enumerate(time_points):
            rows.append({
                "Time_min": t + 1,  # converting 0-index to minute count
                "Cell_ID": cell_id,
                "Raw_Intensity": results[cell_id][i],
                "F0": f0_dict[cell_id],
                "F/F0": f_f0[cell_id][i]
            })
    df = pd.DataFrame(rows)
    csv_save = os.path.join(output_dir, base_name + "_data.csv")
    df.to_csv(csv_save, index=False)
    print("Saved CSV data to", csv_save)

##############################
# 4. Plotting Functions for Analysis (Unchanged)
##############################
def plot_heatmap_from_csv(csv_file, output_folder, contrast_min=None, contrast_max=None,
                          cmap='inferno', figsize=None, font_family="Arial", font_size=12, dpi=300):
    """
    Creates a heatmap of F/Fâ‚€ values from a CSV file.
    Only time points â‰¥ 5 are used. Adjusts the time axis and sorts rows.
    """
    plt.rcParams.update({"font.family": font_family, "font.size": font_size})
    df = pd.read_csv(csv_file)
    
    heatmap_df = df.pivot(index='Cell_ID', columns='Time_min', values='F/F0')
    heatmap_df = heatmap_df.loc[:, heatmap_df.columns >= 5].copy()
    heatmap_df.columns = heatmap_df.columns - 4  # adjust time axis
    
    sort_col = 16  # adjusted time corresponding to original time point 10
    if sort_col in heatmap_df.columns:
        heatmap_df = heatmap_df.sort_values(by=sort_col, ascending=False)
    else:
        print(f"Time_min {sort_col} not found. Sorting by last available time point.")
        heatmap_df = heatmap_df.sort_values(by=heatmap_df.columns.max(), ascending=False)
    
    n_cells, n_time = heatmap_df.shape
    if figsize is None:
        width = max(10, n_time * 0.5)
        height = max(6, n_cells * 0.15)
        figsize = (width, height)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    heatmap_data = heatmap_df.values
    im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, interpolation='nearest',
                   vmin=contrast_min, vmax=contrast_max)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Cell ID")
    ax.set_title("Heatmap of $F/F_0$")
    ax.set_xticks(np.arange(n_time))
    ax.set_xticklabels(heatmap_df.columns, rotation=0)
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label=r"$F/F_0$")
    
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    out_path = os.path.join(output_folder, f"{base_filename}_heatmap_FF0.png")
    fig.savefig(out_path, dpi=dpi)
    plt.show()
    print("Saved heatmap to", os.path.abspath(out_path))
    return heatmap_df

def plot_mean_only(csv_file, output_folder, font_family="Arial", font_size=12, figsize=(8,6),
                   x_min=None, x_max=None, x_tick_interval=None,
                   y_min=None, y_max=None, y_tick_interval=None,
                   save_csv=False, csv_output_path=None, dpi=300):
    """
    Groups the CSV data by time (only time points â‰¥ 5), computes mean F/Fâ‚€,
    adjusts the time axis (original 5 becomes 0, etc.) and plots a line graph.
    """
    plt.rcParams.update({"font.family": font_family, "font.size": font_size})
    df = pd.read_csv(csv_file)
    grouped = df.groupby("Time_min")["F/F0"].mean().reset_index()
    grouped = grouped[grouped["Time_min"] >= 5].copy()
    grouped["Adjusted_Time"] = grouped["Time_min"] - 5
    if save_csv:
        if csv_output_path is None:
            base, ext = os.path.splitext(os.path.basename(csv_file))
            csv_output_path = os.path.join(output_folder, f"{base}_mean_FF0.csv")
        grouped.to_csv(csv_output_path, index=False)
        print("Saved grouped CSV data to", os.path.abspath(csv_output_path))
    
    x = grouped["Adjusted_Time"]
    y = grouped["F/F0"]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x, y, marker="o", label=r"Mean $F/F_0$")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mean $F/F_0$ Intensity")
    ax.set_title("Mean $F/F_0$ Over Time")
    ax.legend()
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()
    ax.set_xlim(x_min, x_max)
    if x_tick_interval is not None:
        xticks = np.arange(x_min, x_max+1, x_tick_interval)
        ax.set_xticks(xticks)
    if y_min is not None or y_max is not None:
        if y_min is None:
            y_min = y.min()
        if y_max is None:
            y_max = y.max()
        ax.set_ylim(y_min, y_max)
    if y_tick_interval is not None:
        yticks = np.arange(y_min, y_max+y_tick_interval, y_tick_interval)
        ax.set_yticks(yticks)
    
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    out_path = os.path.join(output_folder, f"{base_filename}_mean_FF0_line.png")
    fig.savefig(out_path, dpi=dpi)
    plt.show()
    print("Saved mean-only plot to", os.path.abspath(out_path))
    return grouped

def plot_scatter_pre_post_with_ttest(csv_file, output_folder, font_family="Arial", font_size=12, figsize=(8,6),
                                     x_min=None, x_max=None, y_min=0, y_max=8, y_tick_interval=1,
                                     marker_color="blue", marker_alpha=0.7, dpi=300):
    """
    Creates a scatter plot of F/Fâ‚€ vs. SpikePhase (PreSpike vs PostSpike) from the CSV file.
    Performs a two-sample t-test and annotates significance if p < 0.05.
    """
    plt.rcParams.update({"font.family": font_family, "font.size": font_size})
    df = pd.read_csv(csv_file)
    df["SpikePhase"] = np.where(df["Time_min"] <= 9, "PreSpike", "PostSpike")
    phase_mapping = {"PreSpike": 0, "PostSpike": 1}
    df["Phase_Num"] = df["SpikePhase"].map(phase_mapping)
    
    pre_group = df[df["SpikePhase"] == "PreSpike"]["F/F0"]
    post_group = df[df["SpikePhase"] == "PostSpike"]["F/F0"]
    t_stat, p_value = ttest_ind(pre_group, post_group, nan_policy='omit')
    print("t-test: t-statistic = {:.4f}, p-value = {:.4f}".format(t_stat, p_value))
    
    jitter = np.random.uniform(-0.1, 0.1, size=len(df))
    x = df["Phase_Num"] + jitter
    y = df["F/F0"]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(x, y, color=marker_color, alpha=marker_alpha)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["PreSpike", "PostSpike"], rotation=0)
    ax.set_xlabel("")
    ax.set_xlim(x_min if x_min is not None else -0.5,
                x_max if x_max is not None else 1.5)
    ax.set_ylim(y_min, y_max)
    if y_tick_interval is not None:
        yticks = np.arange(y_min, y_max+y_tick_interval, y_tick_interval)
        ax.set_yticks(yticks)
    
    ax.set_ylabel(r"$F/F_0$ Intensity")
    ax.set_title("")
    
    if p_value < 0.05:
        y_range = y_max - y_min
        annotation_y = y_max - 0.08 * y_range
        ax.plot([0, 1], [annotation_y, annotation_y], color="black", lw=1)
        ax.text(0.5, annotation_y + 0.01*y_range, "*", ha="center", va="bottom", 
                color="black", fontsize=font_size+4)
    
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    out_path = os.path.join(output_folder, f"{base_filename}_scatter_FF0_ttest.png")
    fig.savefig(out_path, dpi=dpi)
    plt.show()
    print("Saved scatter plot with t-test to", os.path.abspath(out_path))
    return df

def plot_combined_overlay(csv_file, output_folder, font_family="Arial", font_size=12, dpi=300):
    """
    Creates a combined figure with three subplots:
      - Heatmap of F/Fâ‚€.
      - Mean-only line plot of F/Fâ‚€.
      - Scatter plot of F/Fâ‚€ vs SpikePhase (with t-test annotation).
    """
    heatmap_df = plot_heatmap_from_csv(csv_file, output_folder, contrast_min=1, contrast_max=1.3, cmap='inferno', 
                                       figsize=(12,8), font_family=font_family, font_size=font_size, dpi=dpi)
    mean_grouped = plot_mean_only(csv_file, output_folder, font_family=font_family, font_size=font_size, figsize=(8,6),
                                  save_csv=True, dpi=dpi)
    scatter_df = plot_scatter_pre_post_with_ttest(csv_file, output_folder, font_family=font_family, font_size=font_size, 
                                                  figsize=(8,6), marker_color="orange", marker_alpha=0.4, dpi=dpi)
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)
    
    # Left: Heatmap
    n_cells, n_time = heatmap_df.shape
    heatmap_data = heatmap_df.values
    im = axs[0].imshow(heatmap_data, aspect='auto', cmap='inferno', interpolation='nearest', vmin=1, vmax=1.3)
    axs[0].set_xlabel("Time (min)")
    axs[0].set_ylabel("Cell ID")
    axs[0].set_title("Heatmap of $F/F_0$")
    axs[0].set_xticks(np.arange(n_time))
    axs[0].set_xticklabels(heatmap_df.columns, rotation=0)
    axs[0].set_yticks([])
    fig.colorbar(im, ax=axs[0], label=r"$F/F_0$")
    
    # Middle: Mean-only line plot
    x = mean_grouped["Adjusted_Time"]
    y = mean_grouped["F/F0"]
    axs[1].plot(x, y, marker="o", label=r"Mean $F/F_0$")
    axs[1].set_xlabel("Time (min)")
    axs[1].set_ylabel("Mean $F/F_0$ Intensity")
    axs[1].set_title("Mean $F/F_0$ Over Time")
    axs[1].legend()
    axs[1].set_xlim(x.min(), x.max())
    
    # Right: Scatter plot with t-test annotation
    jitter = np.random.uniform(-0.1, 0.1, size=len(scatter_df))
    x_scatter = scatter_df["Phase_Num"] + jitter
    y_scatter = scatter_df["F/F0"]
    axs[2].scatter(x_scatter, y_scatter, color="orange", alpha=0.4)
    axs[2].set_xticks([0, 1])
    axs[2].set_xticklabels(["PreSpike", "PostSpike"], rotation=0)
    axs[2].set_xlabel("")
    axs[2].set_xlim(-0.5, 1.5)
    axs[2].set_ylabel(r"$F/F_0$ Intensity")
    pre_group = scatter_df[scatter_df["SpikePhase"] == "PreSpike"]["F/F0"]
    post_group = scatter_df[scatter_df["SpikePhase"] == "PostSpike"]["F/F0"]
    t_stat, p_value = ttest_ind(pre_group, post_group, nan_policy='omit')
    if p_value < 0.05:
        y_range = (y_scatter.max() - y_scatter.min())
        annotation_y = y_scatter.max() + 0.05 * y_range
        axs[2].plot([0, 1], [annotation_y, annotation_y], color="black", lw=1)
        axs[2].text(0.5, annotation_y + 0.01*y_range, "*", ha="center", va="bottom", 
                    color="black", fontsize=font_size+4)
    axs[2].set_title("Scatter of $F/F_0$ (t-test p={:.4f})".format(p_value))
    
    fig.tight_layout()
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    combined_out = os.path.join(output_folder, f"{base_filename}_combined_overlay.png")
    fig.savefig(combined_out, dpi=dpi)
    plt.show()
    print("Saved combined overlay figure to", os.path.abspath(combined_out))

##############################
# Main Pipeline Execution (Adapted for TIFF)
##############################
if __name__ == "__main__":
    # Set your main input folder containing timepoint subdirectories.
    main_input_folder = "/home/cephla/Downloads/response_GABA_2025-04-25_15-47-20.017959"  # modify as needed

    # Step 1: Generate z-max projections.
    print("=== Generating z-max projections ===")
    zmax_projection(main_input_folder)
    
    # Define zmax folder (created in step 1).
    zmax_folder = os.path.join(main_input_folder, "zmax")
    
    # Step 2: Run Cellpose segmentation on the z-max TIFFs.
    print("=== Running Cellpose segmentation ===")
    run_cellpose_segmentation_cyto2(
        input_folder=zmax_folder,
        channels=(0, 0),
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        gpu=True  # Set to True if you have a GPU available.
    )
    
    # Step 3: Process each FOV with its corresponding mask.
    # CSV files will be saved in an "output" folder inside the main input folder.
    output_dir = os.path.join(main_input_folder, "output")
    os.makedirs(output_dir, exist_ok=True)
    baseline_times = [5, 6, 7, 8]
    
    # Get all zmax files
    zmax_files = glob.glob(os.path.join(zmax_folder, "*_zmax.tif"))
    if not zmax_files:
        print("No zmax files found in", zmax_folder)
    else:
        for zmax_file in tqdm(zmax_files, desc="Processing FOVs for intensity extraction"):
            base_name = os.path.splitext(os.path.basename(zmax_file))[0]
            fov_key = base_name.replace("_zmax", "")  # e.g., "B2_0"
            
            # Construct the mask file path.
            # Assumes the mask file is named: [base_name]_mask_cyto2.tif in zmax/mask_cyto2.
            mask_filename = base_name + "_mask_cyto2.tif"
            mask_file = os.path.join(zmax_folder, "mask_cyto2", mask_filename)
            if not os.path.exists(mask_file):
                print(f"Mask file not found for {zmax_file} at {mask_file}")
                continue
            process_pair(main_input_folder, mask_file, output_dir, fov_key, 
                        channel="488", baseline_times=baseline_times)
    
    # Step 4: Plot analysis from CSV files.
    analysis_outputs_folder = os.path.join(output_dir, "analysis_outputs")
    os.makedirs(analysis_outputs_folder, exist_ok=True)
    csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found in", output_dir)
    else:
        for csv_file in tqdm(csv_files, desc="Plotting analysis from CSV files"):
            print("=== Processing CSV file:", csv_file, "===")
            plot_heatmap_from_csv(csv_file, analysis_outputs_folder, contrast_min=1, contrast_max=1.3,
                                  cmap='coolwarm', figsize=(12,8), font_family="Arial", font_size=12, dpi=300)
            plot_mean_only(csv_file, analysis_outputs_folder, font_family="Arial", font_size=12, figsize=(8,6),
                           x_min=0, x_max=10, x_tick_interval=1,
                           y_min=0.9, y_max=1.4, y_tick_interval=0.1,
                           save_csv=True, dpi=300)
            plot_scatter_pre_post_with_ttest(csv_file, analysis_outputs_folder, font_family="Arial", font_size=12,
                                             figsize=(4, 8), x_min=-0.5, x_max=1.5,
                                             y_min=0, y_max=9, y_tick_interval=1,
                                             marker_color="orange", marker_alpha=0.4, dpi=300)
            plot_combined_overlay(csv_file, analysis_outputs_folder, font_family="Arial", font_size=12, dpi=300)
