#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import glob


def find_load_env():
    from dotenv import load_dotenv

    files = glob.glob(r"*.env") + ['.env', ]
    files = list(set(files))
    index_env = files.index('.env')
    last_env = files[-1]
    files[-1] = files[index_env]
    files[index_env] = last_env

    for f in files:
        try:
            is_load_env = load_dotenv(dotenv_path=f)
            if is_load_env:
                print(f"Loaded environment variables in file '{f}'")
                return
        except FileNotFoundError as e:
            continue
    else:
        print('No any "*.env" file to load environment variables. Let create a file and export')
