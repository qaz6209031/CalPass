import re

INPUT_FILENAME = 'normalized_1.txt'
OUTPUT_FILENAME = 'normalized_final.txt'

V_COURSE = 'course'
V_PROFESSOR = 'professor'

def main():
   var_pattern = r'\[([^\]]+)\]'
   var_set = set()

   # idea: use dictionary to normalize variable names?
   var_dict = {
      'course': V_COURSE,
      'teacher': V_PROFESSOR,
      'professor': V_PROFESSOR
   }

   in_file = open(INPUT_FILENAME, 'r')
   out_file = open(OUTPUT_FILENAME, 'w')

   for line in in_file:
      for match in re.findall(var_pattern, line):
         match = match.lower()
         var_set.add(match)

      line = line.split('|')
      question = line[1].strip()
      answer = line[2].strip()

      out_file.write(question + '\n')
      out_file.write(answer + '\n')
         
   # test print to see how many variables there are
   print(var_set)
   in_file.close()
   out_file.close()

if __name__ == '__main__':
   main()