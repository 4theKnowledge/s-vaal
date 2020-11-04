"""
Description

@author: Tyler Bikaun
"""

import yaml

def main():
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