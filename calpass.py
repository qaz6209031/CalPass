import requests
import pandas as pd
from bs4 import BeautifulSoup

def main():
	#obtain the content of the URL in HTML
	url = "https://schedules.calpoly.edu/depts_52-CENG_curr.htm"
	page = requests.get(url)

	# Check if the page response correctly
	if page.status_code != 200:
		print('please connect to Cal Poly VPN')
		return

	#Create a soup object that parses the HTML
	soup = BeautifulSoup(page.text,"html.parser")

	personalInfo = []
	# Iterate all the <tr> tag of CS department
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

		# Get the phoneNumber of professor
		phoneNumber = tr.find('td', {'class': "personPhone"})

		# If both name and phone number are not None, append to the personInfo list
		if name and phoneNumber:
			personalInfo.append((name.text, phoneNumber.text))
	
	# print(personalInfo)

if __name__ == "__main__":
	main()