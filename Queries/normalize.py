import re

INPUT_FILENAME = 'normalized_1.txt'
OUTPUT_FILENAME = 'normalized_final.txt'

# V_COURSE = 'course'
# V_PROFESSOR = 'professor'

def main():
   var_pattern = r'\[([^\]]+)\]'
   var_set = set()
   line_set = set()

   # idea: use dictionary to normalize variable names?
   # var_dict = {
   #    'course': V_COURSE,
   #    'teacher': V_PROFESSOR,
   #    'professor': V_PROFESSOR
   # }

   in_file = open(INPUT_FILENAME, 'r')
   out_file = open(OUTPUT_FILENAME, 'w')

   for line in in_file:
      for match in re.findall(var_pattern, line):
         repl = normalize_var(match)
         var_set.add(repl)
         line = line.replace(match, repl, 1)
      
      if line not in line_set:    
         line_set.add(line)
         line = line.split('|')
         question = line[1].strip()
         answer = line[2].strip()

         out_file.write(question + ' | ')
         out_file.write(answer + '\n')
         
   # test print to see how many variables there are
   #print(len(var_set))
   in_file.close()
   out_file.close()
   

def normalize_var(var):
   profssor_key_words = ['professor', 'teacher', 'prof', 'name', 'faculty', 'teachername', 'prof', 'csse-faculty', 'faculty',
                         'prof name', 'poly alias', 'alias', 'statistics-faculty', 'stats-faculty', 'user']
   # To do, come out with more keywords here
   course_key_words = ['coursecode', 'department code', 'roomname', 'coursenumorname', 'coursenumber', 'class', 'class-code',
                       'major', 'csse-course', 'csse-courses', 'prereqs-csse-course', 'prereqs-stats-course', 'stats-course',
                       'stats-section', 'csse-section', 'section']
   bldg_key_words= ['room code', 'location', 'room', 'roomn', 'location', 'csse-faculty-office-location', 'stats-faculty-office-location']
   var = var.lower()
   if var in profssor_key_words:
      var = 'professor'
   elif var in course_key_words:
      var = 'course'
   elif var in bldg_key_words:
      var = 'location'
   return var

if __name__ == '__main__':
   main()