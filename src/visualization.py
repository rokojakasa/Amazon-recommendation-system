import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_preprocess_split

# Example DataFrame
df = pd.read_csv('data/Electronics_5core.csv.gz', compression='gzip')

# Count the number of occurrences of each rating
rating_counts = df['rating'].value_counts().sort_index()

# Plot
plt.figure(figsize=(6,4))
rating_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Rating')
plt.ylabel('Number of Ratings')
plt.title('Distribution of Ratings')
plt.xticks(rotation=0)

# Save plot to file
plt.savefig('results/rating_distribution.png', dpi=300, bbox_inches='tight')  # You can change file format and name

# Optionally show the plot
# plt.show()

val_df = pd.read_csv('data/Electronics_valid.csv.gz', compression='gzip')
print(f"Number of rows: {len(val_df)}")
print(f"Number of unique users: {val_df['user_id'].nunique()}")
