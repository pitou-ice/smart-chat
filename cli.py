import os
import sys
import cmd
from dotenv import load_dotenv
from typing import Final, Iterator
from llama_cpp import Llama
from pymilvus import MilvusClient


load_dotenv()

BOT_NAME: Final[str | None] = os.getenv('BOT_NAME')
BOT_SUBJECT: Final[str | None] = os.getenv('BOT_SUBJECT')
BOT_MODEL_PATH: Final[str | None] = os.getenv('BOT_MODEL_PATH')
DB_EMBEDDING_PATH: Final[str | None] = os.getenv('DB_EMBEDDING_PATH')
DB_COLLECTION_NAME: Final[str | None] = os.getenv('DB_COLLECTION_NAME')

assert BOT_NAME is not None
assert BOT_SUBJECT is not None
assert BOT_MODEL_PATH is not None
assert DB_EMBEDDING_PATH is not None
assert DB_COLLECTION_NAME is not None


class VectorDb:
    '''Milvus client for RAG.'''

    milvus_client = MilvusClient()
    embedding: Final[Llama] = Llama(
        model_path=DB_EMBEDDING_PATH,
        embedding=True,
        verbose=False
    )

    def get_embedding(self, query) -> str:
        try:
            return str(self.embedding.create_embedding(query)['data'][0]['embedding'])
        
        except Exception as e:
            return 'Sorry, embedding failed.'
        
    def query_milvus(self, embedding):
        result_count = 3
  
        result = self.milvus_client.search(
            collection_name=DB_COLLECTION_NAME,
            data=[embedding],
            limit=result_count,
            output_fields=["path", "text"])
  
        list_of_knowledge_base = list(map(lambda match: match['entity']['text'], result[0]))
        list_of_sources = list(map(lambda match: match['entity']['path'], result[0]))
  
        return {
            'list_of_knowledge_base': list_of_knowledge_base,
            'list_of_sources': list_of_sources
        }


class Bot:
    '''Llama.cpp implementation with completion streaming.'''

    prefix: Final[str] = '\r🤖 '
    llm: Final[Llama] = Llama(
        model_path=BOT_MODEL_PATH,
        n_ctx=1024,
        verbose=False
    )

    def print(self, text: str) -> None:
        sys.stdout.write(self.prefix)
        print(text)

    def stream(self, stream: Iterator[str]) -> None:
        sys.stdout.write(self.prefix)
        for word in stream:
            sys.stdout.write(word)
            sys.stdout.flush()
        sys.stdout.write('\n')

    def stream_response(self, query: str, context: str|None=None) -> None:
        prompt = f'''### Instruction:\nYour name is {BOT_NAME}. You are an helpful AI assistant. You only answer questions about {context or BOT_SUBJECT}, or casually chat with the user. Keep your answers short.\n### Input:\n{query}\n### Response:\n'''
        print(prompt)
        try:
            completion_stream = self.llm.create_completion(
                prompt=prompt,
                stop=['###'],
                max_tokens=1024,
                top_k=10,
                temperature=0,
                stream=True,
            )

            assert isinstance(completion_stream, Iterator)
            self.stream((completion['choices'][0]['text'] for completion in completion_stream))

        except Exception as e:
            self.print('Sorry, could you repeat?')


class CLI(cmd.Cmd):
    '''Command line chat interface.'''

    intro: Final[str] = ''' Welcome to \n  ___                    _      ___  _           _   \n / __| _ __   __ _  _ _ | |_   / __|| |_   __ _ | |_ \n \\__ \\| '  \\ / _` || '_||  _| | (__ | ' \\ / _` ||  _|\n |___/|_|_|_|\\__,_||_|   \\__|  \\___||_||_|\\__,_| \\__|\n'''
    prompt: Final[str] = '👤 '
    ruler: Final[str] = ''

    db: Final[VectorDb] = VectorDb()
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
            # embeddings = self.db.get_embedding(query)
            # self.bot.print(embeddings)
            
            # TODO context = self.db.get_context(embeddings)
            self.bot.stream_response(query)

        except Exception as e:
            # TODO define exception types
            self.bot.print('Sorry, I don\'t know what happened.')


if __name__ == '__main__':
    try:
        cli = CLI()
        cli.cmdloop()

    except KeyboardInterrupt as e:
        cli.do_bye()
