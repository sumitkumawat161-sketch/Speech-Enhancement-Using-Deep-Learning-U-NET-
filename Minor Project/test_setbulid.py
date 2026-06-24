import os
import random
import shutil

# -----------------------------
# Folders
# -----------------------------
clean_dir = "data/mixed/clean/"
noisy_dir = "data/mixed/noisy/"

test_clean_dir = "test_set/clean/"
test_noisy_dir = "test_set/noisy/"

# Create output dirs if not exists
os.makedirs(test_clean_dir, exist_ok=True)
os.makedirs(test_noisy_dir, exist_ok=True)

# -----------------------------
# List files
# -----------------------------
clean_files = [f for f in os.listdir(clean_dir) if f.endswith(".wav")]
noisy_files = [f for f in os.listdir(noisy_dir) if f.endswith(".wav")]

# Build dict of noisy files by number prefix
noisy_dict = {}
for f in noisy_files:
    num = f.split("_")[0]  # get initial number
    noisy_dict[num] = f

# Select clean files that have a matching noisy file
matched_clean_files = []
for f in clean_files:
    num = f.split("_")[0]
    if num in noisy_dict:
        matched_clean_files.append(f)

# Randomly select 30 pairs
selected_clean = random.sample(matched_clean_files, min(30, len(matched_clean_files)))

print(f"Selected {len(selected_clean)} file pairs:")
for f in selected_clean:
    print(f)

# Copy clean and noisy files
for clean_fname in selected_clean:
    num = clean_fname.split("_")[0]
    noisy_fname = noisy_dict[num]

    shutil.copy(os.path.join(clean_dir, clean_fname), os.path.join(test_clean_dir, clean_fname))
    shutil.copy(os.path.join(noisy_dir, noisy_fname), os.path.join(test_noisy_dir, noisy_fname))

print("\nDone! Copied clean/noisy file pairs to test_set/.")
