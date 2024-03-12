import os
import sys
import cmd
from dotenv import load_dotenv
from typing import Final, Iterator
from llama_cpp import Llama
from pymilvus import MilvusClient


load_dotenv()

BOT_PROMPT_SYS: Final[str | None] = os.getenv('BOT_PROMPT_SYS')
BOT_MODEL_PATH: Final[str | None] = os.getenv('BOT_MODEL_PATH')
DB_EMBEDDING_PATH: Final[str | None] = os.getenv('DB_EMBEDDING_PATH')
DB_COLLECTION_NAME: Final[str | None] = os.getenv('DB_COLLECTION_NAME')

assert BOT_PROMPT_SYS is not None
assert BOT_MODEL_PATH is not None
assert DB_EMBEDDING_PATH is not None
assert DB_COLLECTION_NAME is not None


# TODO cli tool to populate db
# from langchain_text_splitters import TokenTextSplitter
# text_splitter: Final[TokenTextSplitter] = TokenTextSplitter(chunk_size=50, chunk_overlap=10)
# documents = self.text_splitter.create_documents([DOCUMENT])
# chunks = [doc.page_content for doc in documents]


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

    def stream(self, stream: Iterator) -> None:
        sys.stdout.write(self.prefix)
        for word in stream:
            sys.stdout.write(word['choices'][0]['text'])
            sys.stdout.flush()
        sys.stdout.write('\n')

    def stream_response(self, query: str, context: str) -> None:
        prompt = f"<|system|>{context}\n{BOT_PROMPT_SYS}</s>\n"\
                 f"<|prompt|>{query}</s>\n"\
                  "<|answer|>"
        try:
            completion = self.llm.create_completion(
                prompt=prompt,
                stop=['</s>'],
                max_tokens=134,
                repeat_penalty=1.18,
                stream=True
            )

            assert isinstance(completion, Iterator)
            self.stream(completion)

        except Exception as e:
            self.print('Sorry, could you repeat?')


class CLI(cmd.Cmd):
    '''Command line chat interface.'''

    intro: Final[str] = '\nWelcome to Smart Chat CLI.\n'
    prompt: Final[str] = '👤 '
    ruler: Final[str] = ''

    db: Final[VectorDb] = VectorDb()
    bot: Final[Bot] = Bot()

    def do_bye(self, arg=None) -> None:
        '''\nExits the app\n'''
        self.bot.print('Goodbye')
        exit(0)

    def default(self, query: str) -> None:
        try:
            embeddings = self.db.get_embedding(query)
            self.bot.print(embeddings)
            
            # TODO context = self.db.get_context(embeddings)
            # TODO self.bot.stream_response(query, context)

        except Exception as e:
            # TODO define exception types
            self.bot.print('Sorry, I don\'t know what happened.')


if __name__ == '__main__':
    try:
        cli = CLI()
        cli.cmdloop()

    except KeyboardInterrupt as e:
        cli.do_bye()
