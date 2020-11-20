"""
Module for connections including No-SQL and configuration files
"""

import yaml

def load_config():
    """ Loads configuration file from disk
    """
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)
    return config

class Mongo:
    def __init__(self):
        pass

    def write_experiment(self):
        pass


def main():
    pass

if __name__ == '__main__':
    main()