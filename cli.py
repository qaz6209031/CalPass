import click
import pickle

import calpass
from keywords import load_model, vectorize_query, LABELS

@click.group(invoke_without_command=True)
@click.option('--show_signal', default=False, is_flag=True, help='Display signal for chat bot response')
@click.pass_context
def cli(ctx, show_signal):
   obj = ctx.obj

   model = obj['model']
   vect = obj['vect']

   if show_signal:
      click.echo('Displaying signals')

   while True:
      query = click.prompt('Ask a question')
      query = calpass.normalizeQuery(query)

      bucket = LABELS[model.predict(vectorize_query(vect, query=query))[0]]

      # DEBUG:
      click.echo(f'bucket={bucket}')

      if bucket == 'Professor':
         signal = 'Normal'
         response = calpass.getProfessorInfo(query)
      elif bucket == 'Course':
         signal = 'Normal'
         response = calpass.getCourseInfo(query)
      elif bucket == 'Building':
         signal = 'Normal'
         response = calpass.getBuildingInfo(query)
      elif bucket == 'End':
         signal = 'End'
         click.echo('Goodbye!')
         return

      if response is None or bucket == 'Other':
         signal = 'Unknown'
         response = "I don't understand"
   
      if show_signal:
         click.echo(f'[{signal}]: {response}')
      else:
         click.echo(response)

def main():
   load_model()
   with open('Queries/model.pkl', 'rb') as pkl:
      model = pickle.load(pkl)
   with open('Queries/data.pkl', 'rb') as pkl:
      vect = pickle.load(pkl)
   cli(obj={'model': model, 'vect': vect})

if __name__ == '__main__':
   main()