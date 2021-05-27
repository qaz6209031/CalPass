import requests
import sys
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np

BASE_PATH = 'https://schedules.calpoly.edu/'
HOME_PATH = 'depts_52-CENG_curr.htm'

def main():
	professorDF , urls = constructProfTable()
	courseDF = constructCourseTable(urls)

	print(professorDF)
	print(courseDF)

def constructProfTable():
	profTable, urls = [], set()
	
	#obtain the content of the URL in HTML
	try:
		page = requests.get(BASE_PATH + HOME_PATH)
	except requests.exceptions.RequestException as err:
		sys.exit('Connection error')

	#Create a soup object that parses the HTML
	soup = BeautifulSoup(page.text,"html.parser")

	# Iterate all the <tr> tag of CS department, call parent three times to get back to tr tag 
	for tr in soup.find(text='CENG-Computer Science & Software Engineering').parent.parent.parent.next_siblings:

		# Skip the new line
		if tr == '\n':
			continue

		# Break the loop if we found the next department which is computer engineering
		try:
			department = tr.th.span.text
		except AttributeError:
			department = ""
		if department == 'CENG-Computer Engineering':
			break
		
		# Get the name of the professor
		name = tr.find('td', {'class': "personName"})

		# Get the phone number of professor
		phoneNumber = tr.find('td', {'class': "personPhone"})

		# Get the url of courses
		raw_courses = tr.find_all('td', {'class': "courseName"})
		# courses = set()
		for course in raw_courses:
		# 	courseName = course.text
		# 	courses.add(courseName)
			updateCourseUrl(urls, course)
			
		# If both name and phone number are not None, append to the personInfo list
		if name and phoneNumber:
			if not phoneNumber.text:
				profTable.append((name.text, np.nan))
			else:
				profTable.append((name.text, phoneNumber.text))

	headers = ['Professor Name', 'Phone Number']
	professorDF = pd.DataFrame(profTable, columns = headers)
	return professorDF, urls
	

# Update the urls set to add new course url
def updateCourseUrl(urls, course):
	courseName = course.text
	courseUrl = BASE_PATH + course.find('a')['href']
	urls.add((courseName, courseUrl))
	
def constructCourseTable(urls):
	courseTable = []
	for url in urls:
		courseName = url[0].strip()
		courseUrl = url[1]

		try:
			page = requests.get(courseUrl)
		except requests.exceptions.RequestException as err:
			sys.exit('Connection error')
		
		soup = BeautifulSoup(page.text,"html.parser")
		
		classes = soup.find('tbody').find_all('tr')
		
		for section in classes:
			# if instruction doesn't exit, skip the class
			instructor = section.find('td', {'class': 'personName'}).text
			if instructor == '\xa0':
				continue
			instructorName = section.find('td', {'class': 'personName'}).a['title']
			
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
			locationCapacity = section.find('td', {'class': 'location'}).find_next('td').text
			if locationCapacity == '\xa0':
				locationCapacity = np.nan
			enrollmentCapacity = section.find('td', {'class': 'location'}).find_all_next('td', limit=2)[1].text
			enrolled = section.find('td', {'class': 'location'}).find_all_next('td', limit=3)[2].text
			waitlist = section.find('td', {'class': 'location'}).find_all_next('td', limit=4)[3].text
			drop = section.find('td', {'class': 'location'}).find_all_next('td', limit=5)[4].text

			courseTable.append([courseName, courseSection, courseNumber, courseType, courseDay, startTime, endTime, \
				locationCapacity, enrollmentCapacity, enrolled, waitlist, drop])
			
	# Store the data into panda dataFrame
	headers = ['Course Name', 'Section', 'Course Number', 'Course Type', 'Days', 'Start Time', 'End Time', 'Location Capacity' \
		, 'Enrollment Capacity', 'Enrolled', 'Waitlisted', 'Dropped'] 
	courseDF = pd.DataFrame(courseTable, columns = headers).sort_values(by=['Course Name', 'Section'])
	
	return courseDF

if __name__ == "__main__":
	main()