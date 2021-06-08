from data import getData
from difflib import SequenceMatcher
import nltk
nltk.download('wordnet') # in order to use lemmatizer
from nltk.stem import WordNetLemmatizer
import re

PROFESSOR_TABLE, COURSE_TABLE = getData()
LOCATIONS = set(PROFESSOR_TABLE.office) | set(COURSE_TABLE.Location)

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
	SIMILARITY = 0.8

	# normalize query
	query = normalizeQuery(query)

	# Get the name of professor
	name = getProfName(query)

	if not name:
		return None

	# Pull out data based on name
	filter = (PROFESSOR_TABLE['name'] == name)
	table = PROFESSOR_TABLE.loc[filter]

	# Get the hint of query
	phone = extractEntity(query, PHONE_KEYS, SIMILARITY) 
	officeLocation = extractEntity(query, OFFICE_LOCATION_KEYS, SIMILARITY) 
	department = extractEntity(query, DEPARTMENT_KEYS, SIMILARITY) 
	email = extractEntity(query, EMAIL_KEYS, SIMILARITY) 
	title = extractEntity(query, TITLE_KEYS, SIMILARITY) 
	teach = extractEntity(query, TEACH_KEYS, SIMILARITY) 
	alias = extractEntity(query, ALIAS_KEYS, SIMILARITY) 

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
'''
Logic related to course info
Answered 42% percent of all course related question
'''
def getCourseInfo(query):
	ALL_COURSES = COURSE_TABLE.Course.tolist()
	query = normalizeQuery(query)
	SIMILARITY = 0.8

	WAIT_LIST_KEYS = ['waitlist', 'wait', 'waiting']
	LOCATION_KEYS = ['location', 'where', 'building', 'room']
	NAME_KEYS = ['name', 'called', 'description']
	PROF_KEYS = ['who', 'professor', 'instructor', 'faculty']
	DROP_KEYS = ['drop', 'dropped', 'withdraw']
	START_KEYS = ['start', 'start time']
	END_KEYS = ['end', 'end time']
	DAYS_KEYS = ['day']
	ENROLL_KEYS = ['enroll', 'enrollment', 'capacity', 'available', 'open', 'remaining']

	courseName = getCourseName(query)
	if not courseName:
		return None
	filter = (COURSE_TABLE['Course'] == courseName)
	table = COURSE_TABLE.loc[filter]

	waitlist = extractEntity(query, WAIT_LIST_KEYS, SIMILARITY) 
	location = extractEntity(query, LOCATION_KEYS, SIMILARITY) 
	name = extractEntity(query, NAME_KEYS, SIMILARITY) 
	professor = extractEntity(query, PROF_KEYS, SIMILARITY) 
	drop = extractEntity(query, DROP_KEYS, SIMILARITY) 
	start = extractEntity(query, START_KEYS, SIMILARITY) 
	end = extractEntity(query, END_KEYS, SIMILARITY) 
	day = extractEntity(query, DAYS_KEYS, SIMILARITY) 
	enroll = extractEntity(query, ENROLL_KEYS, SIMILARITY) 

	response = ''
	if waitlist:
		for index, row in table.iterrows():
			response += 'There are ' + row['Waitlisted'] + ' people on the waitlist of ' + courseName + ' section ' + row['Section'] + '. '
	elif location:
		for index, row in table.iterrows():
			response += 'Location of ' + courseName + ' section ' + row['Section'] + ' is ' + row['Location'] + '. '
	elif name:
		response = courseName + ' is ' + table['Name'].iloc[0]
	elif professor:
		profString = ', '.join([prof for prof in list(set(table.Professor.to_list()))])
		response = profString +  ' are teaching ' + courseName
	elif drop:
		for index, row in table.iterrows():
			response += 'There are ' + row['Dropped'] + ' people drop ' + courseName + ' section ' + row['Section'] + '. '
	elif start:
		for index, row in table.iterrows():
			response += 'Start time of ' + courseName + ' section ' + row['Section'] + ' is ' + row['Start Time'] + '. '
	elif end:
		for index, row in table.iterrows():
			response += 'End time of ' + courseName + ' section ' + row['Section'] + ' is ' + row['End Time'] + '. '
	elif day:
		for index, row in table.iterrows():
			response += courseName + ' section ' +  row['Section'] + ' will be taught in ' + row['Days'] + '. '
	elif enroll:
		for index, row in table.iterrows():
			response += courseName + ' section ' +  row['Section'] + ' Enrollement Capacity: ' + row['Enrollment Capacity']+ ' Enrolled: ' + row['Enrolled'] + '. '
	else:
		response = None
	
	return response

# Logic related to building info
def getBuildingInfo():
	query = normalizeQuery(query)
	SIMILARITY = 0.8

	COURSES_KEYS = ['class', 'classes', 'course', 'courses']
	SECTIONS_KEYS = ['section', 'sections']
	PROFESSOR_KEYS = ['professor', 'professors', 'teacher', 'teachers', 'instructor', 'faculty', 'office']
	CAPACITY_KEYS = ['capacity', 'fit', 'size']
	
	buildingNum = getBulidingNum(query)
	if not buildingNum:
		return None

	filteredProfsTable = PROFESSOR_TABLE.loc[PROFESSOR_TABLE['office'] == buildingNum]
	filteredCoursesTable = COURSE_TABLE.loc[COURSE_TABLE['Location'] == buildingNum]

	courses = extractEntity(query, COURSES_KEYS, SIMILARITY) 
	sections = extractEntity(query, SECTIONS_KEYS, SIMILARITY)
	professor = extractEntity(query, PROFESSOR_KEYS, SIMILARITY)
	capacity = extractEntity(query, CAPACITY_KEYS, SIMILARITY)
	
	response = ''
	if courses:
		buildingStr = ', '.join(list(set(filteredCoursesTable.Location.tolist())))
		response = f'The following courses are taught in {buildingNum}, {buildingStr}.'
	elif sections:
		courseSections = {f"{row['Course']}-{row['Section']}" for index, row in filteredCoursesTable.iterrows()}
		buildingStr = ', '.join(courseSections)
		response = f'The following sections are taught in {buildingNum}, {buildingStr}.'
	elif professor:
		for index, row in filteredProfsTable:
			response += f"{row['name']}'s office is at {buildingNum}."
	elif capacity:
		for index, row in filteredCoursesTable:
			response += f"{buildingNum}'s capacity for {row['Course']} is {row['Location Capacity']}."
	else:
		response = None
	
	return response


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

def getProfName(query):
	SIMILARITY = 0.9
	ALL_PROF_NAMES = PROFESSOR_TABLE.name.tolist()
	# Get the name by full name
	name = extractEntity(query, ALL_PROF_NAMES, SIMILARITY)
	if name:
		return name

	# Get the name by last name
	ALL_LAST_NAMES = [name.split(' ')[0] for name in ALL_PROF_NAMES]
	lastName = extractEntity(query, ALL_LAST_NAMES, SIMILARITY)
	for n in ALL_PROF_NAMES:
		if n.startswith(lastName):
			return n

	# Get the name by first name
	ALL_FIRST_NAMES = [name.split(' ')[1] for name in ALL_PROF_NAMES]
	firstName = extractEntity(query, ALL_FIRST_NAMES, SIMILARITY)
	for n in ALL_PROF_NAMES:
		if firstName in n:
			return n
	return None

def getCourseName(query):
	SIMILARITY = 0.9
	ALL_COURSES = COURSE_TABLE.Course.tolist()

	name = extractEntity(query, ALL_COURSES, SIMILARITY)
	if name:
		return name

	return None

def getBulidingNum(query):
	SIMILARITY = 0.9

	buildingNum = extractEntity(query, LOCATIONS, SIMILARITY)
	if buildingNum:
		return buildingNum

	return None

'''
extract entity from the query
keywords are list of words that could be in query
similarity means how close two string are closed to each other (0 ~ 1) 1 means two string are equal
'''
def extractEntity(query, keywords, similarity):
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
			if similar(ngram, key) > similarity:
				return key

	return None

if __name__ == "__main__":
	main()