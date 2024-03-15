import os
import cmd
from typing import Final
from bot import Bot
# from db import VectorDb # Uncomment when RAG is implemented


class CLI(cmd.Cmd):
    '''Command line chat interface.'''

    intro: Final[str] = ''' Welcome to \n  ___                    _      ___  _           _   \n / __| _ __   __ _  _ _ | |_   / __|| |_   __ _ | |_ \n \\__ \\| '  \\ / _` || '_||  _| | (__ | ' \\ / _` ||  _|\n |___/|_|_|_|\\__,_||_|   \\__|  \\___||_||_|\\__,_| \\__|\n'''
    prompt: Final[str] = '👤 '
    ruler: Final[str] = ''

    # db: Final[VectorDb] = VectorDb()
    bot: Final[Bot] = Bot()

    def do_load(self, file: str) -> None:
        '''\nLoad data to Milvus. Usage: load <file>\n'''
        if file:
            self.bot.print(f'Loading {file}...')
        else:
            self.bot.print('Please provide a file. You can tab-complete.')

    def complete_load(self, text: str, line, begidx, endidx) -> list[str]:
        current_dir = os.getcwd()
        files = [f for f in os.listdir(current_dir) if os.path.isfile(f)]
        return [f for f in files if f.__contains__(text)]

    def do_bye(self, arg=None) -> None:
        '''\nExit the app.\n'''
        self.bot.print('Goodbye')
        exit(0)

    def default(self, query: str) -> None:
        try:
            # TODO embeddings = self.db.get_embedding(query)
            # TODO context = self.db.get_context(embeddings)
            self.bot.stream_response(query)

        except Exception as e:
            # TODO define specific exceptions
            self.bot.print('Sorry, I don\'t know what happened.')


if __name__ == '__main__':
    try:
        cli = CLI()
        cli.cmdloop()

    except KeyboardInterrupt as e:
        cli.do_bye()
