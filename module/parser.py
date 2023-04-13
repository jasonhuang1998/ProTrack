import argparse

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type = str, default = 'default', help = 'Config file')
    args = parser.parse_args()
    return args