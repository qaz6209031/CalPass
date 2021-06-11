from data import getData
from difflib import SequenceMatcher
import nltk
# nltk.download('wordnet') # in order to use lemmatizer
from nltk.stem import WordNetLemmatizer
import re
import numpy as np

PROFESSOR_TABLE, COURSE_TABLE = getData()

def main():
	print(normalizeQuery("What courses will professor foaad khosmood teach spring 2021?"))
'''
Once the query get classified as professor, call this function and return proper response
Answered 68% of all professor related quesiton now
Goal: 80%
'''
def getProfessorInfo(query):
	ALL_COURSES = COURSE_TABLE.Course.tolist()
	PHONE_KEYS = ['phone', 'contact', 'call', 'number', 'reach', 'talk', 'get in touch']
	OFFICE_LOCATION_KEYS = ['location', 'where', 'see', 'office']
	DEPARTMENT_KEYS = ['department']
	EMAIL_KEYS = ['email', 'message']
	TITLE_KEYS = ['title', 'type of teacher']
	TEACH_KEYS = ['teach', 'teaching', 'taught', 'instruct', 'courses', 'class', 'section']
	ALIAS_KEYS = ['alias', 'username']
	SIMILARITY = 0.8

	infoObj = {
		'type': 'professor',
		'error': '',
		'target': '',
		'response': ''
	}

	# Get the name of professor
	foundName, name = getProfName(query)

	if not foundName:
		infoObj['error'] = 'failed to find professor name match'
		infoObj['response'] = f'Did you mean {name}?'
		return infoObj
	else:
		infoObj['target'] = name

	# Pull out data based on name
	filter = (PROFESSOR_TABLE['name'] == name)
	table = PROFESSOR_TABLE.loc[filter]

	# Get the hint of query
	phone, _  = extractEntity(query, PHONE_KEYS, SIMILARITY) 
	officeLocation, _ = extractEntity(query, OFFICE_LOCATION_KEYS, SIMILARITY) 
	department, _ = extractEntity(query, DEPARTMENT_KEYS, SIMILARITY) 
	email, _ = extractEntity(query, EMAIL_KEYS, SIMILARITY) 
	title, _ = extractEntity(query, TITLE_KEYS, SIMILARITY) 
	teach, _ = extractEntity(query, TEACH_KEYS, SIMILARITY) 
	alias, _ = extractEntity(query, ALIAS_KEYS, SIMILARITY) 

	if phone and not table['phone'].empty:
		infoObj['response'] = f"Professor {name}'s phone number is {table['phone'].iloc[0]}"
	elif officeLocation and not table['office'].empty:
		infoObj['response'] = f"Professor {name}'s office location is {table['office'].iloc[0]}"
	elif department and not table['department'].empty:
		infoObj['response'] = f"Professor {name}'s department is {table['department'].iloc[0]}"
	elif email and not table['email'].empty:
		infoObj['response'] = f"Professor {name}'s email is {table['email'].iloc[0]}"
	elif title and not table['title'].empty:
		infoObj['response'] = f"Professor {name}'s title is {table['title'].iloc[0]}"
	elif alias and not table['email'].empty:
		infoObj['response'] = f"Professor {name}'s alias is {table['email'].iloc[0].split('@')[0]}"
	elif teach:
		filter = (COURSE_TABLE['Professor'] == name)
		courses = COURSE_TABLE.loc[filter].Course.to_list()
		#Yes / No question
		if any([query.startswith(w) for w in ['does', 'did', 'is']]):
			course, _ = extractEntity(query, ALL_COURSES, SIMILARITY) 
			infoObj['response'] = 'Yes' if course else 'No'
		else:
			courseStr = ' '.join(courses)
			infoObj['response'] = f"Professor {name} is teaching {courseStr} Fall 2021"
	else:
		infoObj['error'] = 'failed to determine question'
		infoObj['response'] = "I don't understand."

	return infoObj
'''
Logic related to course info
Answered 43% percent of all course related question
'''
def getCourseInfo(query):
	ALL_COURSES = COURSE_TABLE.Course.tolist()
	SIMILARITY = 0.8

	WAIT_LIST_KEYS = ['waitlist', 'wait', 'waiting']
	LOCATION_KEYS = ['location', 'where', 'building', 'room', 'virtual']
	NAME_KEYS = ['name', 'called', 'description']
	PROF_KEYS = ['who', 'professor', 'instructor', 'faculty']
	DROP_KEYS = ['drop', 'dropped', 'withdraw', 'dropping']
	START_KEYS = ['start', 'start time']
	END_KEYS = ['end', 'end time']
	DAYS_KEYS = ['day', 'section']
	ENROLL_KEYS = ['enroll', 'enrollment', 'capacity', 'cap', 'available', 'open', 'remaining', 'seats']
	COURSE_NUMBER_KEYS = ['course number', 'class number', 'class code']
	TYPE_KEYS = ['type', 'lab']
	NSECTIONS_KEYS = ['number', 'section', 'opening', 'taught', 'teach', 'offer', 'take']
	TIMES_KEYS = ['time', 'day', 'when', 'MWF', 'TR', 'MTWRF', 'get out']
	CAPACITY_KEYS = ['capacity', 'seat', 'spot', 'enroll']

	infoObj = {
		'type': 'course',
		'error': '',
		'target': '',
		'response': ''
	}

	foundName, courseName = getCourseName(query)
	if not foundName:
		infoObj['error'] = 'failed to find course name match'
		infoObj['response'] = f'Did you mean {courseName}?'
		return infoObj
	else:
		infoObj['target'] = courseName


	filter = (COURSE_TABLE['Course'] == courseName)
	table = COURSE_TABLE.loc[filter]

	waitlist, _ = extractEntity(query, WAIT_LIST_KEYS, SIMILARITY) 
	location, _ = extractEntity(query, LOCATION_KEYS, SIMILARITY) 
	name, _ = extractEntity(query, NAME_KEYS, SIMILARITY) 
	professor, _ = extractEntity(query, PROF_KEYS, SIMILARITY) 
	drop, _ = extractEntity(query, DROP_KEYS, SIMILARITY) 
	start, _ = extractEntity(query, START_KEYS, SIMILARITY) 
	end, _ = extractEntity(query, END_KEYS, SIMILARITY) 
	day, _ = extractEntity(query, DAYS_KEYS, SIMILARITY) 
	enroll, _ = extractEntity(query, ENROLL_KEYS, SIMILARITY) 
	courseNumber, _ = extractEntity(query, COURSE_NUMBER_KEYS, SIMILARITY) 
	types, _ =  extractEntity(query, TYPE_KEYS, SIMILARITY) 
	nsections, _ = extractEntity(query, NSECTIONS_KEYS, SIMILARITY)
	times, _ = extractEntity(query, TIMES_KEYS, SIMILARITY)
	capacity, _ = extractEntity(query, CAPACITY_KEYS, SIMILARITY)
	
	infoObj['response'] = ''
	if waitlist:
		for index, row in table.iterrows():
			infoObj['response'] += f"There are {row['Waitlisted']} people on the waitlist of {courseName} section {row['Section']}.\n"
	elif location:
		for index, row in table.iterrows():
			infoObj['response'] += f"Location of {courseName} section {row['Section']} is {row['Location']}.\n"
	elif name:
		infoObj['response'] = f"{courseName} is {table['Name'].iloc[0]}"
	elif professor:
		profString = ', '.join(list(set(table.Professor.to_list())))
		infoObj['response'] = f"{profString} are teaching {courseName}"
	elif drop:
		for index, row in table.iterrows():
			infoObj['response'] += f"There are {row['Dropped']} people drop {courseName} section {row['Section']}.\n"
	elif start:
		for index, row in table.iterrows():
			infoObj['response'] += f"Start time of {courseName} section {row['Section']} is {row['Days']} {row['Start Time']}.\n"
	elif end:
		for index, row in table.iterrows():
			infoObj['response'] += f"End time of {courseName} section {row['Section']} is {row['Days']} {row['End Time']}.\n"
	elif day:
		for index, row in table.iterrows():
			infoObj['response'] += f"{courseName} section {row['Section']} will be taught in {row['Days']}.\n"
	elif enroll:
		infoObj['response'] += f"Enrollment capacity of {courseName} is {table['Enrollment Capacity'].iloc[0]}. \n"
		for index, row in table.iterrows():
			infoObj['response'] += f"Section {row['Section']} enrolled: {row['Enrolled']}.\n"
	elif courseNumber:
		for index, row in table.iterrows():
			infoObj['response'] += f"{courseName} section {row['Section']} has course number {row['Course Number']}.\n"
	elif types:
		for index, row in table.iterrows():
			infoObj['response'] += f"{courseName} section {row['Section']} is {row['Course Type']}.\n"
	elif nsections:
		infoObj['response'] += f'There are {table.shape[0]} sections of {courseName}.\n'
	elif times:
		infoObj['response'] += f'There are {table.shape[0]} sections of {courseName}. They are offered at the following times:\n'
		for index, row in table.iterrows():
			if row['Days'] == np.nan:
				infoObj['response'] += f"{row['Course']}-{row['Section']}: N/A\n"
			else:
				infoObj['response'] += f"{row['Course']}-{row['Section']}: {row['Days']}, {row['Start Time']}-{row['End Time']}\n"
	elif capacity:
		infoObj['response'] += f'There are {table.shape[0]} sections of {courseName}.\n'
		for index, row in table.iterrows():
			infoObj['response'] += f"{row['Course']}-{row['Section']}:  {row['Enrollment Capacity']} seats, {row['Enrolled']} enrolled, {row['Waitlisted']} waitlisted.\n"
	else:
		infoObj['error'] = 'failed to determine question'
		infoObj['response'] = "I don't understand."
	
	return infoObj

# Logic related to building info
def getBuildingInfo(query):
	SIMILARITY = 0.8

	COURSES_KEYS = ['class', 'classes', 'course', 'courses']
	SECTIONS_KEYS = ['section', 'sections']
	PROFESSOR_KEYS = ['professor', 'professors', 'teacher', 'teachers', 'instructor', 'faculty', 'office', 'teach']
	CAPACITY_KEYS = ['capacity', 'fit', 'size']
	AVAILABILITY_KEYS = ['available', 'occupied', 'free']
	
	infoObj = {
		'type': 'building',
		'error': '',
		'target': '',
		'response': ''
	}

	foundName, buildingNum = getCourseName(query)
	if not foundName:
		infoObj['error'] = 'failed to find building number match'
		infoObj['response'] = f'Did you mean {buildingNum}?'
		return infoObj
	else:
		infoObj['target'] = buildingNum

		

	filteredProfsTable = PROFESSOR_TABLE.loc[PROFESSOR_TABLE['office'] == buildingNum]
	filteredCoursesTable = COURSE_TABLE.loc[COURSE_TABLE['Location'] == buildingNum]

	courses, _ = extractEntity(query, COURSES_KEYS, SIMILARITY) 
	sections, _ = extractEntity(query, SECTIONS_KEYS, SIMILARITY)
	professor, _ = extractEntity(query, PROFESSOR_KEYS, SIMILARITY)
	capacity, _ = extractEntity(query, CAPACITY_KEYS, SIMILARITY)
	availability, _ = extractEntity(query, AVAILABILITY_KEYS, SIMILARITY)
	
	if courses:
		courseStr = ', '.join(list(set(filteredCoursesTable.Course.tolist())))
		infoObj['response'] = f'The following courses are taught in {buildingNum}, {courseStr}.'
	elif sections:
		courseSections = {f"{row['Course']}-{row['Section']}" for index, row in filteredCoursesTable.iterrows()}
		buildingStr = ', '.join(courseSections)
		infoObj['response'] = f'The following sections are taught in {buildingNum}: {buildingStr}.'
	elif professor:
		for index, row in filteredProfsTable.iterrows():
			infoObj['response'] += f"{row['name']}'s office is at {buildingNum}.\n"
	elif capacity:
		for index, row in filteredCoursesTable.iterrows():
			infoObj['response'] += f"{buildingNum}'s capacity for {row['Course']} is {row['Location Capacity']}.\n"
	elif availability:
		if filteredProfsTable.shape[0]:
			infoObj['response'] += f"{buildingNum} is occupied during the following professor(s) office hours: {', '.join(list(filteredProfsTable['name'].tolist()))}\n"
		if filteredCoursesTable.shape[0]:
			infoObj['response'] += f"{buildingNum} is occupied at the following times for classes:\n"
			for index, row in filteredCoursesTable.iterrows():
				infoObj['response'] += f"{row['Course']}-{row['Section']}: {row['Days']}, {row['Start Time']}-{row['End Time']}\n"
	else:
		infoObj['error'] = 'failed to determine question'
		infoObj['response'] = "I don't understand."
	
	return infoObj


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
	found, name = extractEntity(query, ALL_PROF_NAMES, SIMILARITY)
	if found:
		return found, name

	# Get the name by last name
	ALL_LAST_NAMES = [name.split(' ')[0] for name in ALL_PROF_NAMES]
	found, lastName = extractEntity(query, ALL_LAST_NAMES, SIMILARITY)
	if found:
		for n in ALL_PROF_NAMES:
			if n.startswith(lastName):
				return found, n

	# Get the name by first name
	ALL_FIRST_NAMES = [name.split(' ')[1] for name in ALL_PROF_NAMES]
	found, firstName = extractEntity(query, ALL_FIRST_NAMES, SIMILARITY)
	if found:
		for n in ALL_PROF_NAMES:
			if firstName in n:
				return found, n

	return found, name

def getCourseName(query):
	# Higer similarity ratio for matching names
	SIMILARITY = 1
	ALL_COURSES = COURSE_TABLE.Course.tolist()

	return extractEntity(query, ALL_COURSES, SIMILARITY)

def getBuildingNum(query):
	LOCATIONS_RAW = set(PROFESSOR_TABLE.office) | set(COURSE_TABLE.Location)
	# remove nan from locations
	LOCATIONS = [loc for loc in LOCATIONS_RAW if str(loc) != 'nan'] 
	SIMILARITY = 0.9

	return extractEntity(query, LOCATIONS, SIMILARITY)

	

'''
extract entity from the query
keywords are list of words that could be in query
similarity means how close two string are closed to each other (0 ~ 1) 1 means two string are equal
'''
def extractEntity(query, keywords, similarity):
	maxSimilarity = 0
	closestMatch = keywords[0]

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
			curSimilarity = similar(ngram, key)
			if curSimilarity >= similarity:
				return True, key
			elif curSimilarity > maxSimilarity:
				maxSimilarity = curSimilarity
				closestMatch = key

	return False, closestMatch

if __name__ == "__main__":
	main()