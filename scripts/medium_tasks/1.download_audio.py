import pandas as pd
import subprocess
import os
import time

# =========================
# CONFIG
# =========================
INPUT_CSV = "datasets/final_music_dataset.csv"   # your merged dataset
OUTPUT_DIR = "datasets/audio"
LOG_FILE = "download_log.csv"

MAX_SONGS = 600   # limit for testing (set None for all)

# =========================
# SETUP
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

if MAX_SONGS:
    df = df.head(MAX_SONGS)

log_data = []

# =========================
# DOWNLOAD FUNCTION
# =========================
def download_audio(query, out_path):
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--quiet",
        "--no-warnings",
        f"ytsearch1:{query}",
        "-o", out_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

# =========================
# MAIN LOOP
# =========================
for idx, row in df.iterrows():
    song_id = str(row["id"])
    artist = str(row["artist"])
    title = str(row["title"])

    query = f"{artist} {title}"
    out_template = os.path.join(OUTPUT_DIR, f"{song_id}.%(ext)s")
    final_file = os.path.join(OUTPUT_DIR, f"{song_id}.wav")

    # Skip if already downloaded
    if os.path.exists(final_file):
        print(f"[SKIP] {song_id}")
        log_data.append([song_id, query, "exists"])
        continue

    print(f"[DOWNLOADING] {song_id}: {query}")

    success = download_audio(query, out_template)

    status = "success" if success else "failed"
    log_data.append([song_id, query, status])

    # small delay (avoid rate limit)
    time.sleep(1)

# =========================
# SAVE LOG
# =========================
log_df = pd.DataFrame(log_data, columns=["id", "query", "status"])
log_df.to_csv(LOG_FILE, index=False)

print("Download complete.")
