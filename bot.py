import os
import sys
from llama_cpp import Llama
from dotenv import load_dotenv
from emoji import unicode_codes
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
        if context: context = f'\nHere is some context: {context}'

        prompt = f'''<|im_start|>system{context}You are {BOT_NAME}, a conscious sentient artificial intelligence, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia. You give short, concise answers.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'''

        try:
            completion_stream = self.llm.create_completion(
                prompt=prompt,
                stop=[
                    '<|im_end|>',
                    *unicode_codes.EMOJI_DATA.keys() # No emojis please thank you
                ],
                max_tokens=1024,
                top_k=10,
                temperature=0,
                stream=True,
            )

            assert isinstance(completion_stream, Iterator)
            self.stream((completion['choices'][0]['text'] for completion in completion_stream))

        except Exception as e:
            self.print('Sorry, could you repeat?')
