import subprocess
import os
import pandas as pd

repo_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(repo_dir, "util")

feature_a_script = "feature_a.py"
feature_b_script = "feature_b.py"
feature_c_script = "feature_c.py"

#path to csv´s A,B,C
csv_a_csv = ""
csv_b_csv = ""
csv_c_csv = ""

def run_feature(script_name):
    print(f"Running {script_name} ...")
    result = subprocess.run(
        ["python", script_name],
        cwd=util_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
    else:
        print(f"{script_name} completed successfully.")

def clean_mask_name(name):
    base = os.path.splitext(name)[0]
    if base.endswith("_mask"):
        base = base[:-5]
    return base

def merge_csv_files(csv_files):
    dfs = []
    for file in csv_files:
        print(f"Loading {file} ...")
        df = pd.read_csv(file)

        # Rename first column to 'key' for consistent merge
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'key'})

        # Clean the key column
        df['key'] = df['key'].apply(clean_mask_name)

        dfs.append(df)

    print("Merging CSV files on 'key' with inner join (only rows present in all files)...")
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on='key', how='inner')

    return merged_df

if __name__ == "__main__":
    run_feature(feature_a_script)
    run_feature(feature_b_script)
    run_feature(feature_c_script)

    merged = merge_csv_files([csv_a_csv, csv_b_csv, csv_c_csv])

    output_file = os.path.join(repo_dir, "baseline_features.csv")
    merged.to_csv(output_file, index=False)
    print(f"\n✅ Merged CSV saved at {output_file}")
