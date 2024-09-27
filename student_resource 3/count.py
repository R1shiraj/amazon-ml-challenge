import pandas as pd

# Load the dataset (assuming your dataset is named train.csv)
df = pd.read_csv('./dataset/test.csv')

# Find the unique entries in the 'entity_name' column and their counts
unique_entity_names = df['entity_name'].value_counts()

# Print the results
# print(f"Number of unique entries in 'entity_name': {len(unique_entity_names)}")
# print("Counts of each unique entry in 'entity_name':")
print(unique_entity_names)
