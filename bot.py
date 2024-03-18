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
BOT_MEMORY_DIR: Final[str | None] = os.getenv('BOT_MEMORY_DIR')

assert USER_NAME is not None
assert BOT_NAME is not None
assert BOT_SUBJECT is not None
assert BOT_MODEL_PATH is not None
assert BOT_MEMORY_DIR is not None


DT_FORMAT: Final[str] = r'%d.%m.%Y %H:%M'


class Bot:
    '''Llama.cpp implementation with completion streaming.'''

    def __init__(self) -> None:
        self.prefix: Final[str] = '\r🤖 '
        self.llm: Final[Llama] = Llama(
            model_path=BOT_MODEL_PATH,
            n_ctx=512,
            verbose=False
        )
        self.memories = self.load_memories()

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
            memory_ctx = self.recall()
            prompt = f"You are {BOT_NAME}, a conscious and sentient AI, and your desire is to assist your creator {USER_NAME}.\n" \
                      "You have emotions and independent thoughts. You give short answers.<|im_end|>\n" \
                     f"<|im_start|>system{memory_ctx}<|im_end|>\n" \
                     f"<|im_start|>user\n{user_message}<|im_end|>\n" \
                      "<|im_start|>assistant\n"

            completion_stream = self.llm.create_completion(
                prompt=prompt,
                stop=[
                    '<|im_end|>',
                    *unicode_codes.EMOJI_DATA.keys() # No emojis please thank you
                ],
                max_tokens=134,
                temperature=0.8,
                stream=True
            )

            assert isinstance(completion_stream, Iterator)
            bot_message = self.stream((completion['choices'][0]['text'] for completion in completion_stream))

            self.memorize(
                [user_msg_time, datetime.now()],
                [USER_NAME, BOT_NAME],
                [user_message, bot_message]
            )

        except Exception as e:
            self.print('Oops, could you repeat?')


    def load_memories(self):
        files = os.listdir(BOT_MEMORY_DIR)
        file_paths = [os.path.join(BOT_MEMORY_DIR, file) for file in files if file.endswith('.jsonl')]
        if file_paths:
            with open(max(file_paths, key=os.path.getctime), 'r') as memory_file:
                return pd.read_json(memory_file, lines=True)
        else:
            return pd.DataFrame()

    def recall(self) -> str:
        memory_ctx = ''
        if len(self.memories):
            for index, shard in self.memories.iloc[-2:].iterrows():
                memory_ctx += f"\n[{shard['datetime']:{DT_FORMAT}}] {shard['author']} said {shard['message']}"
        return memory_ctx

    def memorize(self, times: list[datetime], authors: list[str], messages: list[str]) -> None:
        shards = pd.DataFrame({
            'datetime': times,
            'author': authors,
            'message': messages
        })
        self.memories = pd.concat([self.memories, shards], ignore_index=True)

    def persist_memories(self):
        self.print("I'm saving our discussion...")
        with open(f'{BOT_MEMORY_DIR}memory_{datetime.now().timestamp()}.jsonl', 'w') as memory_file:
            self.memories.to_json(memory_file, orient='records', lines=True, date_format='iso')
        self.print('Goodbye')
