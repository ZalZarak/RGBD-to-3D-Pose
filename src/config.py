import yaml

path = 'modular_drl_env/RGBDto3Dpose/config/config.yaml'

# Load YAML data from a file
with open(path, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
