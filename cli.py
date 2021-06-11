import click
import pickle

import calpass
from keywords import get_features, load_model, vectorize_query, LABELS

@click.group(invoke_without_command=True)
@click.option('--show_signal', default=False, is_flag=True, help='Display signal for chat bot response')
@click.pass_context
def cli(ctx, show_signal):
   """
   To quit: Ctrl+C, Enter
   """
   obj = ctx.obj

   model = obj['model']
   vect = obj['vect']

   if show_signal:
      click.echo('Displaying signals')

   while True:
      query = click.prompt('Ask a question')
      query = calpass.normalizeQuery(query)

      bucket = LABELS[model.predict(vectorize_query(vect, query=query))[0]]

      if bucket == 'Professor':
         info_obj = calpass.getProfessorInfo(query)
      elif bucket == 'Course':
         info_obj = calpass.getCourseInfo(query)
      elif bucket == 'Building':
         info_obj = calpass.getBuildingInfo(query)
      elif bucket == 'End':
         if show_signal:
            click.echo('[type: end] Goodbye!')
         else:
            click.echo('Goodbye!')
         return
      elif bucket == 'Other':
         info_obj = {
            'type': 'unknown',
            'error': '',
            'target': '',
            'response': 'failed to determine type of question'
         }
   
      signal_str = ''
      if show_signal:
         for info_key in ['type', 'error', 'target']:
            if info_obj[info_key]:
               signal_str += f"[{info_key}: {info_obj[info_key]}] "
   
      click.echo(signal_str + info_obj['response'])

def main():
   load_model()
   with open('Queries/model.pkl', 'rb') as pkl:
      model = pickle.load(pkl)
   with open('Queries/data.pkl', 'rb') as pkl:
      vect = pickle.load(pkl)
   cli(obj={'model': model, 'vect': vect})

if __name__ == '__main__':
   main()