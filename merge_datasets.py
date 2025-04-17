import pandas as pd

df1 = pd.read_csv("student-mat.csv", sep=';')
df2 = pd.read_csv("student-por.csv", sep=';')

df_all = pd.concat([df1, df2], ignore_index=True)

print("Total number of students:", len(df_all))

df_all.to_csv("all_students.csv", index=False)
print(len(df_all))
