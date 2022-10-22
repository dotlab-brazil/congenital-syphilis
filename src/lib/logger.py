import requests
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

from lib.env_var_keys import EnvVarKeys

orig_stdout = sys.stdout


def print_bot(exp_id, message, bot_token_key, bot_chat_id_key):
    load_dotenv()

    telegram_token = os.getenv(bot_token_key)
    telegram_chat_id = os.getenv(bot_chat_id_key)

    message = f"{exp_id}: {message}"
    payload = f'https://api.telegram.org/bot{telegram_token}/sendMessage?chat_id={telegram_chat_id}&text={message}'

    # print(f'Tentando enviar: {payload}')
    response = requests.get(payload)

    return response.json()


def print_to_file(log_file_dir, log_file_name, message):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    f = open(f'{log_file_dir}/ongoing_{log_file_name}', "a", encoding="utf-8")
    f.write(f'[{dt_string}] {message}\n')
    f.close()


class Logger(object):
    def __init__(self, timestamp, exp_id, total_comb, bot_token_key, bot_chat_id_key, bot_enabled=True):
        self.log_file_name = f'{timestamp}_{exp_id}.log'
        self.log_file_dir = os.getenv(EnvVarKeys.LOG_FILE_PATH_KEY.value)

        self.terminal = sys.stdout
        self.log = open(f'{self.log_file_dir}/{self.log_file_name}', "a")
        self.contador = 0
        self.timestamp = timestamp
        self.exp_id = exp_id
        self.total_comb = total_comb
        self.bot_token_key = bot_token_key
        self.bot_chat_id_key = bot_chat_id_key
        self.bot_enabled = bot_enabled

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

        if (message not in "\n" and message.startswith('[CV]')):
            self.contador = self.contador + 1
            info = (
                f"{int(self.contador)}/{self.total_comb} concluídos  ✅")
            message = message + "\n" + info

            if (self.bot_enabled):
                print_bot(self.exp_id, message,
                          self.bot_token_key, self.bot_chat_id_key)
            else:
                print_to_file(self.log_file_dir, self.log_file_name, message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def set_logger(timestamp, exp_id, total_comb, bot_token_key, bot_chat_id_key, bot_enabled=True):
    sys.stdout = Logger(timestamp, exp_id, total_comb,
                        bot_token_key, bot_chat_id_key, bot_enabled)


def reset_logger():
    sys.stdout = orig_stdout


def count_combinations(params, k_fold):
    total = 1

    for key in params[0].keys():
        total = total * len(params[0][key])

    return total * k_fold
