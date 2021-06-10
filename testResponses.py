import pandas as pd
import numpy as np
from calpass import getProfessorInfo, getCourseInfo, getBuildingInfo, normalizeQuery

in_file = open('./Queries/normalized_with_intents.txt', 'r')
out_file = open('./unanswered_queries.txt', 'w')

professorLabel = '0'
courseLabel = '1'
buildingLabel = '2'

profAnswers = profQuestions = courseAnswers = courseQuestions = buildingAnswers = buildingQuestions = 0

for line in in_file:
   lineStripped = line.strip()
   words = lineStripped.split('|')
   question = words[0].lower()
   answer = words[1]
   label = words[2]

   question = question.replace('[professor]', 'Khosmood Foaad')
   question = question.replace('[course]', 'csc 482')
   question = question.replace('[course code]', 'csc 482')
   question = question.replace('[class-number]', 'csc 482')
   question = question.replace('[coursename]', 'csc 482')
   question = question.replace('[location]', '014-0252')
   question = normalizeQuery(question)

   if label == professorLabel:
      profQuestions += 1
      if getProfessorInfo(question):
         profAnswers += 1
      else:
         out_file.write(line)
   elif label == courseLabel:
      courseQuestions += 1
      if getCourseInfo(question):
         courseAnswers += 1
      else:
         out_file.write(line)
   elif label == buildingLabel:
      buildingQuestions += 1
      if getBuildingInfo(question):
         buildingAnswers += 1
      else:
         out_file.write(line)

print(f'{profAnswers / profQuestions * 100}% of Professor questions answered')
print(f'{courseAnswers / courseQuestions * 100}% of Course questions answered')
print(f'{buildingAnswers / buildingQuestions * 100}% of Building questions answered')

