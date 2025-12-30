import pandas as pd
import kagglehub
import os
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------
# Download dataset
# ---------------------------------------
path = kagglehub.dataset_download(
    "wardabilal/spotify-global-music-dataset-20092025"
)

print("Dataset path:", path)

# ---------------------------------------
# Find CSV files
# ---------------------------------------
files = os.listdir(path)
print("Files found:", files)

csv_files = [f for f in files if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV file found")

# Use track_data_final.csv if available
if "track_data_final.csv" in csv_files:
    csv_name = "track_data_final.csv"
else:
    csv_name = csv_files[0]

csv_file = os.path.join(path, csv_name)
print("Using file:", csv_file)

# ---------------------------------------
# Load data
# ---------------------------------------
df = pd.read_csv(csv_file)

print("Original shape:", df.shape)
print(df.info())

# ---------------------------------------
# Data Cleaning
# ---------------------------------------
df.drop_duplicates(inplace=True)

if "popularity" in df.columns:
    df["popularity"].fillna(df["popularity"].mean(), inplace=True)

if "duration_ms" in df.columns:
    df["duration_ms"].fillna(df["duration_ms"].median(), inplace=True)
    df["duration_min"] = df["duration_ms"] / 60000

if "popularity" in df.columns:
    scaler = MinMaxScaler()
    df[["popularity"]] = scaler.fit_transform(df[["popularity"]])

df.columns = df.columns.str.lower().str.replace(" ", "_")

# ---------------------------------------
# Save cleaned data
# ---------------------------------------
df.to_csv("cleaned_spotify_data.csv", index=False)

print("✅ Cleaned file saved: cleaned_spotify_data.csv")
print("Final shape:", df.shape)
