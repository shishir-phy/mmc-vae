import pandas as pd

def create_limited_subset(file_path, target_lang, max_size):
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        if 'language' in df.columns:
            # 1. Filter by language
            # 2. Limit the number of rows using .head()
            subset = df[df['language'] == target_lang].head(max_size)
            
            if not subset.empty:
                actual_count = len(subset)
                print(f"Target language: {target_lang}")
                print(f"Rows extracted: {actual_count} (Limit was {max_size})")
                
                # Save the subset to a new CSV
                output_filename = f"subset_{target_lang}_top_{max_size}.csv"
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

# Example Usage:
# This will extract 300 songs where the language is 'bn' 
# and extract 300 songs where language is 'en'
create_limited_subset('datasets/songs_lyrics.csv', 'bn', 300)
create_limited_subset('datasets/songs_lyrics.csv', 'en', 300)
