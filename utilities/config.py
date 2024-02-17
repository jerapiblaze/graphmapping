import yaml

def ConfigParser(path: str) -> list[dict]:
    config = list()
    with open(path) as f:
        config = list(yaml.load_all(f, Loader=yaml.loader.SafeLoader))
    return config
