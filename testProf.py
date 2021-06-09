import pandas as pd
import numpy as np
from calpass import getProfessorInfo

in_file = open('./Queries/normalized_with_intents.txt', 'r')

rows = []
answered = 0
for line in in_file:
   line = line.strip()
   words = line.split('|')
   question = words[0].lower()
   answer = words[1]
   label = words[2]
   if label != '0':
      continue
   if 'office hour' in question:
      continue
   question = question.replace('[professor]', 'Khosmood Foaad')
   question = question.replace('[course]', 'csc 482')
   question = question.replace('[course code]', 'csc 482')
   response = getProfessorInfo(question)
   if not response:
      response = np.nan
   else:
      answered += 1
   rows.append([question, answer, label, response])

headers = ['question', 'answer', 'label', 'response']
df = pd.DataFrame(rows, columns = headers)

filter = (df['label'] == '0')
table = df.loc[filter].sort_values(by=['response', 'answer'])
table.to_csv('professor.csv', index = False, header = True)

ratio = answered * 100 / table.response.size
print(ratio)