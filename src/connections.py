"""
Module for connections including No-SQL and configuration files
"""

import yaml
from pymongo import MongoClient
from datetime import datetime


def load_config(path=None):
    """ Loads configuration file from disk
    """
    if path is None:
        path = r'config.yaml'

    try:
        with open(path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config
    except Exception as e:
        print(e)


class Mongo:
    """ """
    def __init__(self, collection_name:str):
        self.config = load_config(path=r'conn_config.yaml')
        self.collection_name = collection_name

        if self.collection_name is None:
            raise ValueError

        self.init()

    def init(self):
        """ Initialises database connection to mongodb """
        try:
            # Connection to database
            cluster = MongoClient(self.config['Mongo']['uri'])
            db = cluster[self.config['Mongo']['db']]
            self.collection = db[self.collection_name]
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}: Succesfully connected to mongo db collection - {self.collection_name}')
        except Exception as e:
            print(f'Error connecting to mongo db collection:\n{e}')

    def post(self, data: dict):
        """ Posts to a mongodb collection """
        try:
            self.collection.insert_one(data)
            print('Succesfully posted data to mongo db collection')
        except Exception as e:
            print(f'Error posting data to collection - \n{e}')

    # WORK IN PROGRESS...
    def find(self, field_name, operation=None, value=None):
        """ Searches mongodb collection based on field name and an operation and/or value

        Arguments
        ---------
            field_name : str
                Name of field for searching mongo db collection
            operation : str
                Operation to perform in find operation
            value : float
                Value to apply operation to or search mongo db collection for
        """
        # mongo_find(collection, field_name='name', value='john snow')
        result = self.collection.find({field_name: value}).count()
        print(result)


if __name__ == '__main__':
    # Mongo(collection_name='experiments')
    pass