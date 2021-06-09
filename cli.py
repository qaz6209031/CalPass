import click
import pickle
import click_repl
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer

import calpass
from keywords import get_features

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
   if ctx.invoked_subcommand is None:
      ctx.invoke(chat)

# TODO: use model to determine which bucket from query
def get_info_func(query, model, vectorizer):
   return calpass.getProfessorInfo # dummy return val

@cli.command()
@click.pass_obj
def chat(obj):
   model = obj['model']
   vectorizer = obj['vectorizer']
   should_continue = True

   while should_continue:
      query = click.prompt('Ask a question')
      query = calpass.normalizeQuery(query)
      
      click.echo(f'after normalization, {query}')
      vector = vectorizer.fit_transform([query])
      click.echo(f'after fit transform, {vector}')
      vector = vector.todense()
      click.echo(f'after todense {vector}')
      # click.echo(f'result={model.predict(query[0])}')

      info_func = get_info_func(query, model, vectorizer)

      click.echo(info_func(query))

      # TODO: change later to implement quitting ?
      if True:
         should_continue = False
   
# might not need
# @cli.command()
# def repl():
#    click_repl.repl(click.get_current_context())

if __name__ == '__main__':
   with open('Queries/model.pkl', 'rb') as pkl:
      model = pickle.load(pkl)
   with open('Queries/data.pkl', 'rb') as pkl:
      data = pickle.load(pkl)
      vectorizer = data[2]
   cli(obj={'model': model, 'vectorizer': vectorizer})