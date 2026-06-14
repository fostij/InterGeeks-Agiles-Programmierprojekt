import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json

GERMAN_MODEL_NAME = "bert-base-german-cased"
TARGET_FIELDS = [
        "auto_year", 
        "auto_make", 
        "auto_model",
        "incident_type", 
        "incident_severity", 
        "incident_city", 
        "incident_state",
        "collision_type", 
        "property_damage", 
        "witnesses",
        "authorities_contacted", 
        "number_of_vehicles_involved", 
    ]

