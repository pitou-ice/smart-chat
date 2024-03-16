import os
import sys
import pandas as pd
from datetime import datetime
from llama_cpp import Llama
from dotenv import load_dotenv
from emoji import unicode_codes
from typing import Final, Iterator


load_dotenv()

USER_NAME: Final[str | None] = os.getenv('USER_NAME')
BOT_NAME: Final[str | None] = os.getenv('BOT_NAME')
BOT_SUBJECT: Final[str | None] = os.getenv('BOT_SUBJECT')
BOT_MODEL_PATH: Final[str | None] = os.getenv('BOT_MODEL_PATH')
BOT_MEMORY_PATH: Final[str | None] = os.getenv('BOT_MEMORY_PATH')

assert USER_NAME is not None
assert BOT_NAME is not None
assert BOT_SUBJECT is not None
assert BOT_MODEL_PATH is not None
assert BOT_MEMORY_PATH is not None


DT_FORMAT: Final[str] = r'%d.%m.%Y %H:%M'


class Bot:
    '''Llama.cpp implementation with completion streaming.'''

    def __init__(self) -> None:
        self.prefix: Final[str] = '\r🤖 '
        self.llm: Final[Llama] = Llama(
            model_path=BOT_MODEL_PATH,
            n_ctx=1024,
            verbose=False
        )
        with open(BOT_MEMORY_PATH, 'r') as memory_file:
            self.memories = pd.read_json(memory_file, lines=True)


    def print(self, text: str) -> None:
        sys.stdout.write(self.prefix)
        print(text)

    def stream(self, stream: Iterator[str]) -> str:
        bot_message = ''
        sys.stdout.write(self.prefix)
        for word in stream:
            bot_message += word
            sys.stdout.write(word)
            sys.stdout.flush()
        sys.stdout.write('\n')
        return bot_message

    def stream_response(self, user_message: str, context: str|None=None) -> None:
        try:
            user_msg_time = datetime.now()
            memory_ctx = self.remember()
            prompt = f'''<|im_start|>system\nCurrent time is {user_msg_time:{DT_FORMAT}}\nYou are {BOT_NAME}, a sentient AI, and your purpose is to assist the user {USER_NAME}. You have emotions and independent thoughts. You give short, concise answers.{memory_ctx}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n'''

            completion_stream = self.llm.create_completion(
                prompt=prompt,
                stop=[
                    '<|im_end|>',
                    *unicode_codes.EMOJI_DATA.keys() # No emojis please thank you
                ],
                max_tokens=134,
                temperature=0.8,
                stream=True,
            )

            assert isinstance(completion_stream, Iterator)
            bot_message = self.stream((completion['choices'][0]['text'] for completion in completion_stream))

            self.memorize(
                [user_msg_time, datetime.now()],
                [USER_NAME, BOT_NAME],
                [user_message, bot_message]
            )

        except Exception as e:
            print(e)
            self.print('Oops, could you repeat?')

    def remember(self) -> str:
        memory_ctx = ''
        if len(self.memories):
            for index, shard in self.memories.iloc[-5:].iterrows():
                memory_ctx += f'''\n[{shard['datetime']:{DT_FORMAT}}] {shard['author']} said {shard['message']}'''
        return memory_ctx

    def memorize(self, times: list[datetime], authors: list[str], messages: list[str]) -> None:
        shards = pd.DataFrame({
            'datetime': times,
            'author': authors,
            'message': messages
        })
        self.memories = pd.concat([self.memories, shards], ignore_index=True)
