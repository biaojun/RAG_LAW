import json

def read_config(config_path):
    """Read a JSON configuration file and return a dictionary"""
    try:
        # Open and read the file
        with open(config_path, 'r', encoding='utf-8') as f:
            # Parse JSON content into a Python dictionary
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise Exception(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise Exception(f"Invalid JSON format in configuration file: {config_path}")


# for test
if __name__ == "__main__":
    config = read_config()
    print(config["ZHIPU_MAX_BATCH_SIZE"])