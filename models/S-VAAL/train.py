"""
Trainer for generalisation of S-VAAL model.

@author: Tyler Bikaun
"""

import yaml

class Trainer():
    """ """
    def __init__(self):
        pass


    def train(self):
        """ 
        Sequentially trains S-VAAL
        """


        





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