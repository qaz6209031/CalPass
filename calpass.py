import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_PATH = 'https://schedules.calpoly.edu/'
HOME_PATH = 'depts_52-CENG_curr.htm'

def main():
	# List of tuple, first is name, second is phone number, third is courses teached (set)
	personalInfo= []
	# List of tuple, first is courseName, second is url
	urls = set()

	#obtain the content of the URL in HTML
	page = requests.get(BASE_PATH + HOME_PATH)

	# Check if the page response correctly
	if page.status_code != 200:
		print('Error coneecting to page')
		return

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

		# Get the courses of professor
		raw_courses = tr.find_all('td', {'class': "courseName"})
		courses = set()
		for course in raw_courses:
			courseName = course.text
			courses.add(courseName)
			updateCourseUrl(urls, course)
			
		# If both name and phone number are not None, append to the personInfo list
		if name and phoneNumber:
			personalInfo.append((name.text, phoneNumber.text, courses))

	constructCourseTable(urls)

# Update the urls set to add new course url
def updateCourseUrl(urls, course):
	courseName = course.text
	courseUrl = BASE_PATH + course.find('a')['href']
	urls.add((courseName, courseUrl))

def constructCourseTable(urls):
	courseTable = []
	for url in urls:
		courseName = url[0]
		courseUrl = url[1]
		page = requests.get(courseUrl)

		# Check if the page response correctly
		if page.status_code != 200:
			print('Error coneecting to page')
			return
		
		soup = BeautifulSoup(page.text,"html.parser")
		
		classes = soup.find('tbody').find_all('tr')
		
		for c in classes:
			# if instruction doesn't exit, skip the class
			instructor = c.find('td', {'class': 'personName'}).text
			if instructor == '\xa0':
				continue
			instructorName = c.find('td', {'class': 'personName'}).a['title']
			
			courseSection = c.find('td', {'class': 'courseSection'}).text
			courseNumber = c.find('td', {'class': 'courseClass'}).text
			courseType = c.find('td', {'class': 'courseType'}).span.text
			days = c.find('td', {'class': 'courseDays'}).text
			if days == '\xa0':
				days = ''
			startTime = c.find('td', {'class': 'startTime'}).text
			if startTime == '\xa0':
				startTime = ''
			endTime = c.find('td', {'class': 'endTime'}).text
			if endTime == '\xa0':
				endTime = ''
			locationCapacity = c.find('td', {'class': 'location'}).find_next('td').text
			if locationCapacity == '\xa0':
				locationCapacity = ''
			enrollmentCapacity = c.find('td', {'class': 'location'}).find_all_next('td', limit=2)[1].text
			enrolled = c.find('td', {'class': 'location'}).find_all_next('td', limit=3)[2].text
			waitlist = c.find('td', {'class': 'location'}).find_all_next('td', limit=4)[3].text
			drop = c.find('td', {'class': 'location'}).find_all_next('td', limit=5)[4].text

			courseTable.append((courseName, courseSection, courseNumber, courseType, days, startTime, endTime, \
				locationCapacity, enrollmentCapacity, enrolled, waitlist, drop))
			
	print(courseTable)	

if __name__ == "__main__":
	main()