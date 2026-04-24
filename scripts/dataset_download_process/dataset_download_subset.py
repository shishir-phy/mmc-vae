import gdown
import shutil
import pandas as pd
import os

# Create the directory
os.makedirs('datasets', exist_ok=True)

# URL
url = 'https://drive.google.com/file/d/10U40H89cjaI_bCLj0qX8sz-7o8I0uayY/view?usp=drive_link'

# output filename
output = 'datasets/song_lyrics.csv.zip'

gdown.download(url, output, quiet=False)

## Extract
print("Extracting zip file ... ... ...", end=" ")
shutil.unpack_archive("datasets/song_lyrics.csv.zip", "datasets")
print("done")


def count_single_language(file_path, target_lang):
    try:
        df = pd.read_csv(file_path)

        if 'language' in df.columns:
            filtered_df = df[df['language'].str.lower() == target_lang.lower()]

            count = len(filtered_df)

            if count > 0:
                print(f"Count for '{target_lang}': {count}")
            else:
                print(f"No entries found for language: {target_lang}")
        else:
            print("Error: The column 'language' was not found.")

    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Count 
count_single_language('datasets/song_lyrics.csv', 'en')
count_single_language('datasets/song_lyrics.csv', 'bn')


################ Create Subsets ################################
def create_limited_subset(file_path, target_lang, max_size):
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        if 'language' in df.columns:
            subset = df[df['language'] == target_lang].head(max_size)
            
            if not subset.empty:
                actual_count = len(subset)
                print(f"Target language: {target_lang}")
                print(f"Rows extracted: {actual_count} (Limit was {max_size})")
                
                # Save the subset to a new CSV
                output_filename = f"datasets/subset_{target_lang}_top_{max_size}.csv"
                subset.to_csv(output_filename, index=False)
                print(f"File saved successfully as: {output_filename}")
            else:
                print(f"No data found for language: {target_lang}")
        else:
            print("Error: 'language' column not found.")

    except FileNotFoundError:
        print("Error: The CSV file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


### subset 
subset_length=300
# This will extract 300 songs where the language is 'bn' 
# and extract 300 songs where language is 'en'
create_limited_subset('datasets/song_lyrics.csv', 'bn', 300)
create_limited_subset('datasets/song_lyrics.csv', 'en', 300)

