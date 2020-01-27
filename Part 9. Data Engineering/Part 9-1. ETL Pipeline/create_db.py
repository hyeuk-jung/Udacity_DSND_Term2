# Code to create a SQLite database from a Python dataframe

import pandas as pd
import re
import sqlite3

# Load the dataset you want to store into database
df = pd.read_csv('../data/population_data.csv', skiprows = 4)

# Initialize and connect to the database file
conn = sqlite3.connect('../data/population_data.db')

# Replace ' ' (space) in the column names to '_' (underscore) and
# Drop the unnecessary column 'Unnamed:_62'
columns = []
for col in df.columns:
    col = col.replace(' ', '_')
    columns.append(col)
    
df.columns = columns
df.drop(['Unnamed:_62'], axis = 1, inplace = True)

# Save the dataframe into database file
df.to_sql("population_data", conn, if_exists = "replace")
