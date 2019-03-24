#!/usr/bin/env python

import yaml


class Params(object):
    def __init__(self, yml_f, conf_name):
        with open(yml_f) as fp:
            params = yaml.load(fp)[conf_name]
            object.__setattr__(self, '_params', params)

    def __getattr__(self, key):
        return self._params[key]

    def __setattr__(self, key, value):
        self._params[key] = value
