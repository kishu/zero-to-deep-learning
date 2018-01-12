import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('../data/titanic-train.csv')
# DataFrame.head(n=5)
# Returns first n rows
df.head()
# DataFrame.info(verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None)
# Concise summary of a DataFrame
df.info()
# Generate various summary statistics, excluding NaN values.
# DataFrame.describe(percentiles=None, include=None, exclude=None)
df.describe()

#
# Indexing
#
# DataFrame.iloc
# Purely integer-location based indexing for selection by position.
df.iloc[3]
# DataFrame.loc
# Purely label-location based indexer for selection by label.
df.log[0:4, 'Ticket']
df['Ticket'].head()
df[['Embarked', 'Ticket']].head()

#
# Selections
#
df[df['Age'] > 70] # returns DataFrame
df['Age'] > 70 # returns T/F each item
df.query("Age > 70") # returns DataFrame
df[df['Age'] == 11]) & (df['SibSp'] == 5)] # returns DataFrame
df[(df.Age == 11) | (df.Sibsp == 5)] # returns DataFrame
df.query('(Age == 11) | (SibSp == 5)')

#
# Unique Values
#
df['Embarked'].unique() # ['S', 'C', 'Q', nan]

#
# Sorting
#
df.sort_values('Age', ascending = False).head()

#
# Aggeregations
#
df.groupby('Pclass', 'Survied'])['PassengerId'].count()
df.pivot_table(index='Pclass',
                columns='Survied',
                values='PassengerId',
                aggfunc='count')
# DataFrame.corr(method='pearson', min_periods=1)
# Compute pairwise correlation of columns, excluding NA/null values
df.corr()['Survide'].sort_values()

