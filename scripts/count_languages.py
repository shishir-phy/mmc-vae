import pandas as pd

def count_languages(file_path):
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Check if 'language' column exists
        if 'language' in df.columns:
            # Use value_counts() to get counts for each unique value
            # dropna=True is the default (excludes missing values)
            lang_counts = df['language'].value_counts()
            
            print("Language Counts:")
            # Iterate through the Series and print in the requested format
            for lang, count in lang_counts.items():
                print(f"- {lang} {count}")
        else:
            print("Error: The column 'language' was not found in the CSV.")
            
    except FileNotFoundError:
        print("Error: The file was not found. Please check the file path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the script
count_languages('datasets/songs_lyrics.csv')
