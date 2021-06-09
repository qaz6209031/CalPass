import pandas as pd
import numpy as np
from calpass import getCourseInfo

in_file = open('./Queries/normalized_with_intents.txt', 'r')

rows = []
answered = 0
for line in in_file:
   line = line.strip()
   words = line.split('|')
   question = words[0].lower()
   answer = words[1]
   label = words[2]

   question = question.replace('[course]', 'csc 482')
   question = question.replace('[course code]', 'csc 482')
   question = question.replace('[class number]', 'csc 482')
   response = getCourseInfo(question)

   if response:
      answered += 1

   rows.append([question, answer, label, response])

headers = ['question', 'answer', 'label', 'response']
df = pd.DataFrame(rows, columns = headers)

filter = (df['label'] == '1')
table = df.loc[filter].sort_values(by=['response', 'answer'])
table.to_csv('course.csv', index = False, header = True)

ratio = answered * 100 / table.response.size
print(ratio)