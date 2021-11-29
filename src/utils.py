import os

def mkdir_conditional(dirname: str):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
