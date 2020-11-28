"""
Module for connections including No-SQL and configuration files
"""

import yaml
import sys, traceback
from pymongo import MongoClient
from datetime import datetime
import uuid

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
            post_data = {"created": datetime.now()}
            post_data.update(data)
            self.collection.insert_one(post_data)
            print(f'{datetime.now()}: Succesfully posted data to mongo db collection')
        except Exception as e:
            print(f'Error posting data to collection - \n{e}\n')
            traceback.print_exc(file=sys.stdout)
            
    def post_exp_init(self, exp_name=None, settings=None, runs=None):
        """ Posts initial data for start of experiment """
        try:
            # Make placeholder for run data
            run_ph = {str(run): None for run in range(1,runs+1,1)}
            
            id = str(uuid.uuid4())
            post_data = {"_id": id,
                         "created": datetime.now(),
                         "name": exp_name,
                         "info": {"start timestamp": None,
                                  "finish timestamp": None,
                                  "run time": None},
                         "settings": settings,
                         "results": run_ph,
                         "samples": run_ph,
                         "predictions": {
                             "results": run_ph,
                             "statistics": run_ph}
                         }
            self.collection.insert_one(post_data)
            print(f'{datetime.now()}: Succesfully posted data to mongo db collection')
            return id
        
        except Exception as e:
            print(f'Error posting data to collection - \n{e}\n')
            traceback.print_exc(file=sys.stdout)
    
    def post_exp_data(self, id, run, field=None, sub_field=None, data=None):
        """ Adds data to an experiment via post """
        try:
            self.collection.update_one({"_id": id},
                                       {"$set": {f'{field}.{run}' if sub_field is None else f'{field}.{sub_field}.{run}': data}},
                                       False,
                                       True
                                       )
            print(f'{datetime.now()}: Succesfully updated data in mongo db collection')
        except Exception as e:
            print(f'Error posting data to collection - \n{e}\n')
            traceback.print_exc(file=sys.stdout)
    
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
        result = self.collection.find_one({field_name: value})
        print(result)


if __name__ == '__main__':
    mongo = Mongo(collection_name='experiments')
    
    id = mongo.post_exp_init(runs=5)
    
    field = 'results'
    run = '1'
    data = 'hello world'
    mongo.post_exp_data(id=id, run=run, field=field, data=data)
    
    run = '2'
    data = 'goodbye world'
    mongo.post_exp_data(id=id, run=run, field=field, data=data)
    
    run = '1'
    field = 'predictions'
    sub_field = 'results'
    data = 'tyler was here'
    mongo.post_exp_data(id=id, run=run, field=field, sub_field=sub_field, data=data)