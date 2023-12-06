import yaml

# Replace with your config
path = 'config/config.yaml'

# Load YAML data from a file
with open(path, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
