import os
import datetime


now = datetime.datetime.now
logs_folder = 'logs'


def GetLoggerPath(name: str):
    return f"{logs_folder}/{name}.log"


class Logger():
    def __init__(self, path):
        self.path = path
        self.file = open(self.path, 'a', buffering=1)

    def close(self):
        self.file.close()

    def Write(self, *values):
        splitter = '\n'
        self.file.write(f"{now()}:\n{splitter.join(map(str, values))}")


def GetNextLoggerName():
    loggers_names = os.listdir(f'{logs_folder}')
    indexes = [-1]
    for name in loggers_names:
        try:
            indexes.append(int(name.split('_')[1].split('.')[0]))
        except:
            continue
    return f"{logs_folder}/logs_{max(indexes) + 1}.log"


class LoggerCreate():
    def __init__(self, name: str | Logger | None):
        self.need_close = True
        if type(name) is str:
            self.logger_path = GetLoggerPath(name)
        elif type(name) is Logger:
            self.logger = name
            self.logger_path = name.file.name
            self.need_close = False
        elif name is None:
            self.logger_path = GetNextLoggerName()

    def __enter__(self):
        if self.need_close:
            self.logger = Logger(self.logger_path)
        return self.logger

    def __exit__(self, *args):
        if self.need_close:
            print(f"Write at logger: {self.logger_path}")
            self.logger.close()


def WriteAtLogger(logger = None, *values):
    if logger:
        logger.Write(map(str, values))
    else:
        print(*values)