from data import getData
from difflib import SequenceMatcher
import re

SIMILARITY = 0.8
PROFESSOR_TABLE, COURSE_TABLE = getData()
ALL_PROF_NAMES = PROFESSOR_TABLE.name.tolist()
PHONE_KEYS = ['phone', 'contact', 'call', 'number']
OFFICE_HOUR_KEYS = []


def main():
	getProfessorInfo("What is professor Lupo Christopher's number ")
	pass

def getProfessorInfo(query):
	# normalize query
	query = normalizeQuery(query)
	print('Query after normalization: ', query)
	name = extractEntity(query, ALL_PROF_NAMES)
	print('Professor name: ', name)

	# Pull out data based on name
	filter = (PROFESSOR_TABLE['name'] == name)
	table = PROFESSOR_TABLE.loc[filter]

	# Get the hint of query
	phone = extractEntity(query, PHONE_KEYS) 
	if phone:
		response = 'Professor ' + name + "'s phone number is " + PROFESSOR_TABLE['phone'].iloc[0]
	else:
		response = "Sorry I don't recognize this question, please try another one"

	print(response)

# Logic related to course info
def getCourseInfo():
	pass

# Logic related to building info
def getBuildingInfo():
	pass

def normalizeQuery(query):
	# lowercase
	query = query.lower()
	# remove punctuation
	query = re.sub(r'[^\w\s]', '', query)
	
	return query
'''
extract entity from the query
keywords are list of words that could be in query
'''
def extractEntity(query, keywords):
	# Find the similarity between two strings, return value 0 ~ 1
	def similar(a, b):
		return SequenceMatcher(None, a, b).ratio()

	# Split the query into ngrams
	def ngrams(query, n):
		query = query.split(' ')
		output = []
		for i in range(len(query)-n+1):
			output.append(query[i:i+n])
		return [' '.join(o) for o in output]

	for key in keywords:
		length = len(key.split())
		ngramList = ngrams(query, length)
		for ngram in ngramList:
			if similar(ngram, key) > SIMILARITY:
				return key

	return None

if __name__ == "__main__":
	main()