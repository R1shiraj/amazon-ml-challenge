import pandas as pd

# Load the existing output CSV with missing indices
output_df = pd.read_csv('height_predictions.csv')  # Replace with your actual output file name

# Ensure index column is correctly interpreted
output_df['index'] = output_df['index'].astype(int)

# Create a DataFrame with a full range of indices (assuming max index is 100,000)
max_index = 131287  # Replace this with the actual maximum index if known
full_index_df = pd.DataFrame({'index': range(max_index + 1)})

# Merge the existing output with the full index DataFrame
# This will ensure all indices from 0 to max_index are present
merged_df = pd.merge(full_index_df, output_df, on='index', how='left')

# Fill missing predictions with empty strings
merged_df['prediction'] = merged_df['prediction'].fillna('')

# Save the updated DataFrame to a new CSV file
merged_df.to_csv('updated_output.csv', index=False)

print("CSV file with all indices and missing predictions filled has been saved.")
