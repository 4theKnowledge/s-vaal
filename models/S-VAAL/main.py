"""
Main script which orchestrates model initialisation, model training, etc.

@author: Tyler Bikaun
"""

import yaml



def main(config):
    """
    Does something someday...
    
    """

    # will do something one day
    print(config)

if __name__ == '__main__':
    try:
        with open(r'config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)

    main(config)

