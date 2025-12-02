import pandas as pd
import matplotlib.pyplot as plt

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
plt.show()
