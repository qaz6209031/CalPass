from data import getData
from difflib import SequenceMatcher
import nltk
nltk.download('wordnet') # in order to use lemmatizer
from nltk.stem import WordNetLemmatizer
import re

# similarity between two string
SIMILARITY = 0.8
PROFESSOR_TABLE, COURSE_TABLE = getData()

def main():
	
	pass

'''
Once the query get classified as professor, call this function and return proper response
Answered 68% of all professor related quesiton now
Goal: 80%
'''
def getProfessorInfo(query):
	ALL_COURSES = COURSE_TABLE.Course.tolist()
	PHONE_KEYS = ['phone', 'contact', 'call', 'number', 'reach', 'talk', 'get in touch']
	OFFICE_LOCATION_KEYS = ['location', 'where', 'see']
	DEPARTMENT_KEYS = ['department']
	EMAIL_KEYS = ['email', 'message']
	TITLE_KEYS = ['title', 'type of teacher']
	TEACH_KEYS = ['teach', 'teaching', 'taught', 'instruct', 'courses']
	ALIAS_KEYS = ['alias', 'username']

	# normalize query
	query = normalizeQuery(query)

	# Get the name of professor
	name = getName(query)

	if not name:
		return None

	# Pull out data based on name
	filter = (PROFESSOR_TABLE['name'] == name)
	table = PROFESSOR_TABLE.loc[filter]

	# Get the hint of query
	phone = extractEntity(query, PHONE_KEYS) 
	officeLocation = extractEntity(query, OFFICE_LOCATION_KEYS) 
	department = extractEntity(query, DEPARTMENT_KEYS) 
	email = extractEntity(query, EMAIL_KEYS) 
	title = extractEntity(query, TITLE_KEYS) 
	teach = extractEntity(query, TEACH_KEYS) 
	alias = extractEntity(query, ALIAS_KEYS) 

	if phone:
		response = 'Professor ' + name + "'s phone number is " + table['phone'].iloc[0]
	elif officeLocation:
		response = 'Professor ' + name + "'s office location is " + table['office'].iloc[0]
	elif department:
		response = 'Professor ' + name + "'s department is " + table['department'].iloc[0]
	elif email:
		response = 'Professor ' + name + "'s email is " + table['email'].iloc[0]
	elif title:
		response = 'Professor ' + name + "'s title is " + table['title'].iloc[0]
	elif alias:
		response = 'Professor ' + name + "'s alia is " + table['email'].iloc[0].split('@')[0]
	elif teach:
		filter = (COURSE_TABLE['Professor'] == name)
		courses = COURSE_TABLE.loc[filter].Course.to_list()
		#Yes / No question
		if any([query.startswith(w) for w in ['does', 'did', 'is']]):
			course = extractEntity(query, ALL_COURSES) 
			response = 'Yes' if course else 'No'
		else:
			courseStr = ' '.join(courses)
			response = 'Professor ' + name + " is teaching " + courseStr + ' Fall 2021'
	else:
		response = None
	
	print(name)
	print(query)
	print(response)

	return response

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

	# Lemmatize
	wnl = WordNetLemmatizer()
	lemmatizedQuery = [wnl.lemmatize(word) for word in query.split()]
	query = ' '.join(lemmatizedQuery)

	return query

def getName(query):
	ALL_PROF_NAMES = PROFESSOR_TABLE.name.tolist()
	# Get the name by full name
	name = extractEntity(query, ALL_PROF_NAMES)
	if name:
		return name

	# Get the name by last name
	ALL_LAST_NAMES = [name.split(' ')[0] for name in ALL_PROF_NAMES]
	lastName = extractEntity(query, ALL_LAST_NAMES)
	for n in ALL_PROF_NAMES:
		if n.startswith(lastName):
			return n

	# Get the name by first name
	ALL_FIRST_NAMES = [name.split(' ')[1] for name in ALL_PROF_NAMES]
	firstName = extractEntity(query, ALL_FIRST_NAMES)
	for n in ALL_PROF_NAMES:
		if firstName in n:
			return n
	return None

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