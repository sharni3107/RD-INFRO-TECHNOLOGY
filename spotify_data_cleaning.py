import pandas as pd
import kagglehub
import os
from sklearn.preprocessing import MinMaxScaler

path = kagglehub.dataset_download("wardabilal/spotify-global-music-dataset-20092025")

print("Dataset path:", path)

files = os.listdir(path)
print("Files found:", files)

csv_files = [f for f in files if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV file found")

if "track_data_final.csv" in csv_files:
    csv_name = "track_data_final.csv"
else:
    csv_name = csv_files[0]

csv_file = os.path.join(path, csv_name)
print("Using file:", csv_file)

df = pd.read_csv(csv_file)

print("Original shape:", df.shape)
print(df.info())

df.drop_duplicates(inplace=True)

if "track_popularity" in df.columns:
    df["track_popularity"].fillna(df["track_popularity"].mean(), inplace=True)

if "track_duration_ms" in df.columns:
    df["track_duration_ms"].fillna(df["track_duration_ms"].median(), inplace=True)
    df["duration_min"] = df["track_duration_ms"] / 60000

if "track_popularity" in df.columns:
    scaler = MinMaxScaler()
    df[["track_popularity"]] = scaler.fit_transform(df[["track_popularity"]])

df.columns = df.columns.str.lower().str.replace(" ", "_")

df.to_csv("cleaned_spotify_data.csv", index=False)

print("✅ Cleaned file saved: cleaned_spotify_data.csv")
print("Final shape:", df.shape)

df_sorted = df.sort_values(by="track_popularity", ascending=False)
small_table = df_sorted[["track_name", "artist_name", "track_popularity"]].head(20)

print("\n--- Top 20 Tracks by Popularity ---\n")
print(small_table)

small_table.to_csv("top20_tracks.csv", index=False)
print("\n✅ Top 20 tracks saved: top20_tracks.csv")


