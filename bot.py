import os
import sys
from llama_cpp import Llama
from dotenv import load_dotenv
from typing import Final, Iterator


load_dotenv()

BOT_NAME: Final[str | None] = os.getenv('BOT_NAME')
BOT_SUBJECT: Final[str | None] = os.getenv('BOT_SUBJECT')
BOT_MODEL_PATH: Final[str | None] = os.getenv('BOT_MODEL_PATH')

assert BOT_NAME is not None
assert BOT_SUBJECT is not None
assert BOT_MODEL_PATH is not None


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
