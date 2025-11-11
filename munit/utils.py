# coding = utf-8
import yaml

# Ths is a comment.
# load yaml file
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

