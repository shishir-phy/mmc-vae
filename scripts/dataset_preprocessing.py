import pandas as pd

bn = pd.read_csv("subset_bn_top_300.csv")
en = pd.read_csv("subset_en_top_300.csv")

## Inspect both subset
print(bn.head())
print(en.head())

print(bn.columns)
print(en.columns)

## Fix Differences

# Add language labels
bn["language"] = "Bangla"
en["language"] = "English"

# Ensure same column names
bn = bn.rename(columns={
    "song_name": "title",
    "singer": "artist"
})

en = en.rename(columns={
    "track": "title",
    "artist_name": "artist"
})

## Remove Duplicates
bn = bn.drop_duplicates(subset=["title", "artist"])
en = en.drop_duplicates(subset=["title", "artist"])

## Remove missing
bn = bn.dropna(subset=["title", "artist"])
en = en.dropna(subset=["title", "artist"])


## After Cleaning
n = min(len(bn), len(en))

bn = bn.sample(n, random_state=42)
en = en.sample(n, random_state=42)

print("After Cleaning:")
print("English: ", len(en))
print("Bangla: ", len(bn))


## Merge datasets
df = pd.concat([bn, en], ignore_index=True)

# Create unique ID
df["id"] = range(len(df))

print("After Merge:")
print(df["language"].value_counts())


## Check Lyrics length
df["lyrics_len"] = df["lyrics"].apply(lambda x: len(x.split()))

df = df[df["lyrics_len"] > 30]  # remove short lyrics

print(df.isnull().sum())


## Save final dataset
df.to_csv("final_music_dataset.csv", index=False)

## Dataset plot
#df["language"].value_counts().plot(kind="bar")

ax = df["language"].value_counts().plot(kind="bar")
fig = ax.get_figure()
fig.savefig("plots/final_dataset_by_language.png")


