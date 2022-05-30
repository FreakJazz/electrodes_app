import os
from fastapi import FastAPI
from pydantic import BaseModel
from airtable import airtable
from dotenv import load_dotenv, dotenv_values
import requests
from schemas import Electrodes

load_dotenv()

# Config table data from airtable database
AIRTABLE_BASE_ID=os.environ.get("BASE_ID")
AIRTABLE_API_KEY=os.environ.get("API_KEY")
AIRTABLE_TABLE_NAME=os.environ.get("TABLE_NAME")

endpoint = f'https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}'

endpoint_get = f'https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/ELECTRODES'

# 
app = FastAPI(title= 'Test-app',
                description= 'CRUD API WITH AIRTABLE AND FAST API',
                version='1.0.0')

# data class
class Data(BaseModel):
    name: str


    class Config:
        orm_mode = True

@app.post('/api/electrodes/cross_validation/')
async def analisis_cross_validation(info: Electrodes):
    print(info)

    headers = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json"
    }
    data = { 
    "records": [
    {
    "fields": {
        "Date": info.solver, 
        "E1":  str,
        "E2":  str,
        "E3":  str,
        "E4":  str,
        "E5":  str,
        "E6":  str,
        "E7":  str,
        "E8":  str,
        "E9":  str,
        "E10": str,
        "E11": str,
        "E12": str,
        "E13": str,
        "E14": str,
        "E15": str,
        "E16": str,
        "E17": str,
        "E18": str,
        "E19": str,
        "E20": str,
        "E21": str,
        "E22": str,
        "E23": str,
        "E24": str,
        "E25": str,
        "E26": str,
        "E27": str,
        "E28": str,
        "E29": str,
        "E30": str,
        "E31": str,
        "E32": str
    }
    }
    ]
    }
            

    r = requests.post(endpoint, json=data, headers=headers)


    return user