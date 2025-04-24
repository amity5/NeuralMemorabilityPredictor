import pandas as pd
import h5py
import numpy as np
import time

def load_memorability_data():
    """Load memorability data."""
    filename = r"D:\THINGS\THINGS_Memorability_Scores.csv"
    memo_df = pd.read_csv(filename)
    return memo_df


def load_category_data():
    """Load category data."""
    filename = r"D:\THINGS\Concept_to_category_linking.csv"
    category_df = pd.read_csv(filename)
    return category_df


start_time = time.time()  # Start timing

# Subject ID
subject_id = "sub-03"  # Change this to "sub-01", "sub-02", etc., as needed

# Enable TEST MODE
TEST_MODE = False  # Set to False for full data


# File paths
base_path = r"D:\THINGS\betas_csv\betas_csv"
voxel_metadata_file = rf"{base_path}\{subject_id}_VoxelMetadata.csv"
stimulus_metadata_file = rf"{base_path}\{subject_id}_StimulusMetadata.csv"
response_data_file = rf"{base_path}\{subject_id}_ResponseData.h5"

# Output file paths
suffix = "_TEST" if TEST_MODE else ""
voxel_response_file = rf"{base_path}\{subject_id}_VoxelResponses{suffix}.csv"
voxel_roi_mapping_file = rf"{base_path}\{subject_id}_Voxel_ROI_Mapping{suffix}.csv"


# Load voxel metadata and stimulus metadata
voxel_metadata = pd.read_csv(voxel_metadata_file)
stimulus_metadata = pd.read_csv(stimulus_metadata_file)  # Load original file

# Load memorability and category data
memo_df = load_memorability_data()
category_df = load_category_data()

# Print the first few rows of the voxel column
# print(stimulus_metadata.columns)  # To see available column names
# print(stimulus_metadata.head())   # To inspect the dataframe


# Define specific ROIs manually
regions_of_interest = {
    "lLOC": "Left Lateral Occipital Complex",
    "rLOC": "Right Lateral Occipital Complex",
    "lFFA": "Left Fusiform Face Area",
    "rFFA": "Right Fusiform Face Area",
    "lPPA": "Left Parahippocampal Place Area",
    "rPPA": "Right Parahippocampal Place Area",
    "lEBA": "Left Extrastriate Body Area",
    "rEBA": "Right Extrastriate Body Area",
    "IT": "Inferotemporal Cortex",
    # Memory-related ROIs
    "glasser-PHA1": "Parahippocampal Area 1",
    "glasser-PHA2": "Parahippocampal Area 2",
    "glasser-PHA3": "Parahippocampal Area 3",
    "glasser-EC": "Entorhinal Cortex",
    "glasser-PreS": "Presubiculum",
    "glasser-H": "Hippocampus",
    "glasser-ProS": "Prosubiculum",
    "glasser-PeEc": "Perirhinal Cortex",
    "glasser-RSC": "Retrosplenial Cortex",  # (Glasser Atlas)
    "lRSC": "Left Retrosplenial Cortex",
    "rRSC": "Right Retrosplenial Cortex",
    # Limbic/Orbitofrontal Areas
    "glasser-25": "Subgenual Cortex",
    "glasser-s32": "Superior Anterior Cingulate Cortex",
    "glasser-pOFC": "Posterior Orbitofrontal Cortex",
    # Prefrontal Memory Areas
    "glasser-10r": "Right Anterior Prefrontal Cortex",
    "glasser-10v": "Ventral Anterior Prefrontal Cortex",
    "glasser-9m": "Medial Prefrontal Cortex",
    "glasser-46": "Dorsolateral Prefrontal Cortex",
    "glasser-9-46d": "Dorsomedial Prefrontal Cortex",
}


# Load response data
with h5py.File(response_data_file, "r") as h5_file:
    response_data = h5_file["ResponseData"]["block0_values"][()]  # Beta weights
    voxel_ids = h5_file["ResponseData"]["axis1"][()]  # Voxel IDs

# Ensure voxel count matches
assert response_data.shape[0] == len(voxel_metadata), "Mismatch in voxel count!"

# Create a dictionary to track voxel-to-ROI mapping
voxel_roi_map = {}

# Iterate over manually selected ROIs and find voxel indices
for short_name, full_name in regions_of_interest.items():
    if short_name in voxel_metadata.columns:
        for voxel_index in voxel_metadata[voxel_metadata[short_name] == 1].index:
            if voxel_index not in voxel_roi_map:
                voxel_roi_map[voxel_index] = []
            voxel_roi_map[voxel_index].append(
                full_name
            )  # Add all ROIs voxel belongs to

# Build map from voxel_id to its row index
voxel_id_to_index = {row["voxel_id"]: i for i, row in voxel_metadata.iterrows()}

# Get only voxel_ids that appear in both voxel metadata and ROI map
roi_voxel_ids = [vid for vid in voxel_id_to_index if vid in voxel_roi_map]

# Apply TEST_MODE limit after ROI filtering
if TEST_MODE:
    roi_voxel_ids = roi_voxel_ids[:10]

# Final used IDs and their indices in the original matrix
used_voxel_ids = roi_voxel_ids
used_voxel_indices = [voxel_id_to_index[vid] for vid in used_voxel_ids]


#  Filter voxel_roi_map accordingly
filtered_voxel_roi_map = {
    vid: voxel_roi_map[vid] for vid in used_voxel_ids
}


# Convert voxel-to-ROI mapping to DataFrame
voxel_mapping_df = pd.DataFrame(
    {
        "Voxel ID": list(filtered_voxel_roi_map.keys()),
        "ROIs": [
            ", ".join(roi_list) for roi_list in filtered_voxel_roi_map.values()
        ],  # Multiple ROIs stored per voxel
    }
)

# Save updated voxel-to-ROI mapping
voxel_mapping_df.to_csv(voxel_roi_mapping_file, index=False)

# Define HDF5 output file
hdf5_response_file = voxel_response_file.replace(".csv", ".h5")  # Change file extension

# Open HDF5 file to store voxel responses
with h5py.File(hdf5_response_file, "w") as h5f:
    # Create datasets
    num_stimuli = len(stimulus_metadata["stimulus"].unique())
    num_voxels = len(used_voxel_ids)

    # Create datasets for data
    h5f.create_dataset("voxel_responses", shape=(num_stimuli, num_voxels), dtype="float32", compression="gzip")
    h5f.create_dataset("stimulus_names", shape=(num_stimuli,), dtype=h5py.string_dtype())
    h5f.create_dataset("concepts", shape=(num_stimuli,), dtype=h5py.string_dtype())
    h5f.create_dataset("memorability_scores", shape=(num_stimuli,), dtype="float32")
    h5f.create_dataset("category_labels", shape=(num_stimuli,), dtype=h5py.string_dtype())
    h5f.create_dataset("voxel_ids", data=np.array(used_voxel_ids))  # ✅ store voxel ID mapping once


    # Process each stimulus in a memory-efficient way
    for i, (stimulus_name, group) in enumerate(stimulus_metadata.groupby("stimulus")):
        trial_type = group["trial_type"].iloc[0]
        trial_indices = group.index.values

        # Process response data row-by-row instead of holding everything in memory
        responses = response_data[:, trial_indices]

        if trial_type == "test":
            responses = responses.mean(axis=1)  # Average across test trials
        else:
            responses = responses[:, 0]  # Take first response

        # Get memorability and category
        memorability_score = memo_df.loc[memo_df["image_name"] == stimulus_name, "cr"].values
        category_label = category_df.loc[category_df["concept"] == group["concept"].iloc[0], "category_label"].values

        # Handle missing values
        memorability_score = memorability_score[0] if len(memorability_score) > 0 else np.nan
        category_label = category_label[0] if len(category_label) > 0 else "Unknown"

        # Save
        h5f["stimulus_names"][i] = stimulus_name
        h5f["concepts"][i] = group["concept"].iloc[0]
        h5f["memorability_scores"][i] = memorability_score
        h5f["category_labels"][i] = str(category_label)
        h5f["voxel_responses"][i, :] = responses[used_voxel_indices]

    

print(f"Voxel responses saved to {hdf5_response_file}")
print(f"Voxel-to-ROI mapping saved to {voxel_roi_mapping_file}")

'''


hdf5_file_path = r"D:\THINGS\betas_csv\betas_csv\sub-01_VoxelResponses.h5"

# Open the HDF5 file and list all datasets
with h5py.File(hdf5_file_path, "r") as h5f:
    datasets = list(h5f.keys())
    print("📦 Datasets in HDF5 file:")
    for dset in datasets:
        print(f"- {dset}: shape = {h5f[dset].shape}, dtype = {h5f[dset].dtype}")

    # Try reading a few entries if voxel_ids exist
    if "voxel_ids" in h5f:
        voxel_ids_sample = h5f["voxel_ids"][:10]
        print("\n🔍 Sample voxel IDs:", voxel_ids_sample)

    if "voxel_responses" in h5f:
        responses_sample = h5f["voxel_responses"][:5, :5]
        print("\n🧠 Sample voxel responses (5x5):\n", responses_sample)
    '''
