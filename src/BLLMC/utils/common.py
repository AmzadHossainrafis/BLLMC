"""
licence: MIT
author: Amzad Hossain Rafi
date: 2026-05-12
version: 0.1.0



change log :

    v1.0.0 : Initial release
    branch : main

    date : 2026-05-12
    add: configuration file loading -->> load_project_config

"""

import yaml


def read_config(config_path):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
