#!/usr/bin/env python
# coding: utf-8

# In[4]:

import json
def open_json_file(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
        return data
