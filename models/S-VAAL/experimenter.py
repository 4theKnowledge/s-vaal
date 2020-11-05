"""
Runs batch based experiments on models

@author: Tyler Bikaun
"""

# TODO:
# - Add experiment naming, directory creation, parameter setting, etc.

import yaml

def main(config):
    """"""
    # do something someday
    pass


if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    main(config)