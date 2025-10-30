import json
import os
from datetime import datetime

def load_config():
    default = {"cooldown": 1.5, "alert_sound": True}
    try:
        with open('config.json') as f:
            return {**default, **json.load(f)}
    except:
        return default

def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f)