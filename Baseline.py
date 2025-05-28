import pandas as pd
import subprocess
import os


repo_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(repo_dir, "util")

# Feature scripts
feature_a_script = "feature_a.py"
feature_b_script = "feature_b.py"
feature_c_script = "feature_c.py"

# ABC CSVs
csv_a_csv = " " #ouput path to Asymmetry csv
csv_b_csv = " " # output path to Border csv
csv_c_csv = " " # output path to Color Csv


def run_feature(script_name):
    result = subprocess.run(
        ["python", script_name],
        cwd=util_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"❌ Error running {script_name}:\n{result.stderr}")
    else:
        print(f" {script_name} completed.")


def clean_mask_name(name):
    base = os.path.splitext(name)[0]
    if base.endswith("_mask"):
        base = base[:-5]
    return base


def merge_csv_files(csv_files):
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'key'})
        df['key'] = df['key'].apply(clean_mask_name)
        dfs.append(df)
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on='key', how='inner')
    
    return merged_df


def create_classification_dataset():
    baseline_features_path = os.path.join(repo_dir, "baseline_features.csv")
    metadata_path = " "
    
    
    baseline_df = pd.read_csv(baseline_features_path)
    metadata_df = pd.read_csv(metadata_path)
    
    
    metadata_df['img_id'] = metadata_df['img_id'].str.replace(".png", "", regex=False)

    
    diagnosis_map = {
        'MEL': 1, 'BCC': 1, 'SCC': 1,
        'NEV': 0, 'SEB': 0, 'ACK': 0
    }
    metadata_df['label'] = metadata_df['diagnostic'].map(diagnosis_map)

    
    merged_df = pd.merge(
        baseline_df,
        metadata_df[['img_id', 'label']],
        left_on='key',
        right_on='img_id',
        how='inner'
    )
    merged_df.drop(columns=['img_id'], inplace=True)

    
    output_path = os.path.join(repo_dir, "Baseline_classification.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"Classification dataset saved at {output_path}")
    return merged_df


if __name__ == "__main__":
    print("=" * 50)
    print("STEP 1: GENERATING ABC FEATURES")
    print("=" * 50)

    run_feature(feature_a_script)
    run_feature(feature_b_script)
    run_feature(feature_c_script)

    print("\nSTEP 2: MERGING CSV FILES")
    merged = merge_csv_files([csv_a_csv, csv_b_csv, csv_c_csv])

    output_file = os.path.join(repo_dir, "baseline_features.csv")
    merged.to_csv(output_file, index=False)
    print(f"Baseline features CSV saved at {output_file}")

    print("\nSTEP 3: CREATING CLASSIFICATION DATASET")
    create_classification_dataset()
