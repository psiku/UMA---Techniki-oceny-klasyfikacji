import pandas as pd
import matplotlib.pyplot as plt

file_path = "datasets/wdbc.data"
label_column = "Diagnosis"

df = pd.read_csv(file_path, delimiter=",")

print(df.head(5))
class_counts = df[label_column].value_counts()
class_percentages = df[label_column].value_counts(normalize=True) * 100

print("Class distribution (counts):")
print(class_counts)
print("\nClass distribution (percentages):")
print(class_percentages.round(2))

plt.figure(figsize=(8, 5))
class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
