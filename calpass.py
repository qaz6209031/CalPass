from data import getData

def main():
	professorTable, courseTable = getData()
	

"""Get the answer of professor related questions
@type data: pandas DataFrame
@param data: DataFrame contains all professor's information
@type name: str
@param name: professor's name
@type hint: str
@param hint: information about professor could be 'phone number', 'email', 'department'
@rtype: str
@returns: response based on the question
"""
def getProfessorInfo(data, name, hint):
	# Retrieve the data from professor table
	filter = (data['name'] == name)
	table = data.loc[filter]
	hint = hint.lower()
	result = table[hint].iloc[0]
	
	response = 'Professor ' + name + "'s " + hint + ' is ' + result
	return response
	
if __name__ == "__main__":
	main()