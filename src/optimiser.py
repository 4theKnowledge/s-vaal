"""
Runs optimisation of hyperparameters of neural networks. This will be fleshed out to use Bayesian optimisation.

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

    # Seeds
    np.random.seed(config['Utils']['seed'])
    torch.manual_seed(config['Utils']['seed'])

    main(config)