import sys
import requests
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup


BASE_PATH = 'https://schedules.calpoly.edu/'
ENGINEERING_DEPT_URL = BASE_PATH + 'depts_52-CENG_2218.htm'
SCIENCE_DEPT_URL = BASE_PATH + 'depts_76-CSM_2218.htm'

def getData():
   professorDF , courseUrls = constructProfTable()
   courseDF = constructCourseTable(courseUrls)
   return professorDF, courseDF

def constructProfTable():
	profTable, courseUrls = [], set()
	constructProfessor(ENGINEERING_DEPT_URL, 'Computer Science', courseUrls, profTable)
	constructProfessor(SCIENCE_DEPT_URL, 'Statistics', courseUrls, profTable)

	headers = ['name', 'phone', 'title', 'email', 'office', 'department']
	professorDF = pd.DataFrame(profTable, columns = headers)
	return professorDF, courseUrls
	
def constructProfessor(webUrl, department, courseUrls, profTable):
	#obtain the content of the URL in HTML
	try:
		page = requests.get(webUrl)
	except requests.exceptions.RequestException as err:
		sys.exit('Connection error')
	
	#Create a soup object that parses the HTML
	soup = BeautifulSoup(page.text,"html.parser")

	# Iterate all the <tr> tag of department, call parent three times to get back to tr tag 
	for tr in soup.find(text=re.compile(department)).parent.parent.parent.next_siblings:

		# Skip the new line
		if tr == '\n':
			continue
		
		# Stop when tr finds next department
		if tr.find('span',  {'class': 'subjectDiv'}):
			break
		
		if tr.find('th', {'class': 'header'}):
			continue
		
		# Get the name of the professor
		name = tr.find('td', {'class': 'personName'}).text.replace(',', '').lower()

		# Get the phone number of professor
		phoneNumber = tr.find('td', {'class': 'personPhone'}).text if tr.find('td', {'class': 'personPhone'}).text != '' else np.nan

		# Get the tile of professor
		title = tr.find('td', {'class': 'personTitle'}).text.lower()
		if 'chair' in title:
			title = 'chair'
		elif 'instr' in title:
			title = 'instructor faculty'
		elif 'lecturer' in title:
			title = 'lecturer'

		# Get the location of professor
		office = tr.find('td', {'class': 'personLocation'}).text if tr.find('td', {'class': 'personLocation'}).text != '\xa0' else np.nan

		# Get the email of professor
		email = tr.find('td', {'class': 'personAlias'}).text + '@calpoly.edu'

		# Get the url of courses
		raw_courses = tr.find_all('td', {'class': 'courseName'}) 
		for course in raw_courses:
			updateCourseUrl(courseUrls, course)
		
		profTable.append((name, phoneNumber, title, email, office, department))

# Update the courseUrls set to add new course url
def updateCourseUrl(courseUrls, course):
	courseName = course.text
	# If the course doesn't exist
	if course.find('a') is None:
		return
	courseUrl = BASE_PATH + course.find('a')['href']
	courseUrls.add((courseName, courseUrl))
	
def constructCourseTable(courseUrls):
	courseTable = []
	for url in courseUrls:
		course = url[0].strip().lower()
		courseUrl = url[1]

		try:
			page = requests.get(courseUrl)
		except requests.exceptions.RequestException as err:
			sys.exit('Connection error')
		
		soup = BeautifulSoup(page.text,"html.parser")
		
		courseName = soup.h1.text
		classes = soup.find('tbody').find_all('tr')
		
		for section in classes:
			# if instruction doesn't exit, skip the class
			instructor = section.find('td', {'class': 'personName'}).text
			if instructor == '\xa0':
				continue
			instructorName = section.find('td', {'class': 'personName'}).a['title'].replace(',', '').lower()

			courseSection = section.find('td', {'class': 'courseSection'}).text
			courseNumber = section.find('td', {'class': 'courseClass'}).text
			if courseNumber == '****':
				courseNumber = np.nan
			courseType = section.find('td', {'class': 'courseType'}).span.text
			courseDay = section.find('td', {'class': 'courseDays'}).text
			if courseDay == '\xa0':
				courseDay = np.nan
			startTime = section.find('td', {'class': 'startTime'}).text
			if startTime == '\xa0':
				startTime = np.nan
			endTime = section.find('td', {'class': 'endTime'}).text
			if endTime == '\xa0':
				endTime = np.nan
			location = section.find('td', {'class': 'location'}).text
			if location == '\xa0':
				location = np.nan
			locationCapacity = section.find('td', {'class': 'location'}).find_next('td').text
			if locationCapacity == '\xa0':
				locationCapacity = np.nan
			enrollmentCapacity = section.find('td', {'class': 'location'}).find_all_next('td', limit=2)[1].text
			enrolled = section.find('td', {'class': 'location'}).find_all_next('td', limit=3)[2].text
			waitlist = section.find('td', {'class': 'location'}).find_all_next('td', limit=4)[3].text
			drop = section.find('td', {'class': 'location'}).find_all_next('td', limit=5)[4].text

			courseTable.append([course, courseName, courseSection, courseNumber, courseType, courseDay, startTime, endTime, \
				instructorName, location, locationCapacity, enrollmentCapacity, enrolled, waitlist, drop])
			
	# Store the data into panda dataFrame
	headers = ['Course', 'Name', 'Section', 'Course Number', 'Course Type', 'Days', 'Start Time', 'End Time', 'Professor', 'Location',\
		 'Location Capacity', 'Enrollment Capacity', 'Enrolled', 'Waitlisted', 'Dropped'] 
	courseDF = pd.DataFrame(courseTable, columns = headers).sort_values(by=['Course', 'Section'])
	
	return courseDF

def matchProfName(name, names):
	first_name = name.split()[0]
	for n in names:
		full_first_name = n.split()[0]
		if full_first_name == first_name:
			return n
	return first_name
