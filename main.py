"""
Authors: Jazmin Rodriguez Bermeo and Ivan Aguas
Description: 
This is a backend for EEG data with IA two methods 
use framework FASTAPI and AIRTABLE DATABASE
"""
import asyncio
from datetime import datetime
import time
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from scipy.io import loadmat
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from schemas.electrodes import Electrodes
from schemas.cross_validation import CrossValidation
from schemas.percentage import Porcentage

load_dotenv()

# Config table data from airtable database
AIRTABLE_BASE_ID=os.environ.get("BASE_ID")
AIRTABLE_API_KEY=os.environ.get("API_KEY")
AIRTABLE_TABLE_NAME=os.environ.get("TABLE_NAME")

endpoint = f'https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}'
# https://api.airtable.com/v0/appMxptalwhbXopM9/ELECTRODES

endpoint = f'https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/ELECTRODES'
endpoint_get = f'https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/ELECTRODES'


# 
app = FastAPI(title= 'IA EEG',
                description= 'GET ELECTRODES IA',
                version='1.0.0')

# data class
class Data(BaseModel):
    name: str

    class Config:
        orm_mode = True

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f'{process_time:0.1f} sec')
    return response
##############################CROSS VALIDATION##################################
async def get_val_electrodes(cross: dict, k: int, accuracy: dict):
    limit = 0.05 
    count = 2
    folder_data = 'sessions'
    se = ['s1.mat', 's2.mat', 's3.mat', 's4.mat']
    n_correct_final = np.zeros((4, 20))
    for s in range(4):
        session_test = se[s]
        session_training = se[:s] + se[s+1:]
        X_train = np.empty((32, 32, 0))
        y_train = np.empty((0))
        X_test = np.empty((32, 32, 0))
        y_test = np.empty((0))
        scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        n_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        print(k)
        # Data Training
        for entry in session_training:
            file_selected = folder_data + '/' + entry
            count = count + 1
            if file_selected is not 'session_training/.DS_Store':
                # print(file_selected)
                data_session = loadmat(file_selected, squeeze_me=True, struct_as_record=False)
                for i in range(6):
                    session = data_session['runs'][i]
                    X_train = np.concatenate((X_train, session.x), axis=2)
                    y_train = np.concatenate((y_train, session.y), axis=0)
        winsorized_x = winsorize(X_train[k,:,:], limits=limit) 
        normalize_x = stats.zscore(winsorized_x, axis=None)
        X_train = normalize_x.T
        if cross['solver'] == 'svd':
            clf = LinearDiscriminantAnalysis(solver=cross['solver'], tol=cross['tol'], store_covariance=False)
        else:
            clf = LinearDiscriminantAnalysis(solver=cross['solver'], tol=cross['tol'], store_covariance=False, shrinkage=cross['shrinkage'])
        clf.fit(X_train, y_train)
        file_selected = folder_data + '/' + session_test
        data_session = loadmat(file_selected, squeeze_me=True, struct_as_record=False)
        for i in range(6):
            session = data_session['runs'][i]
            X_test = session.x
            winsorized_x_test = winsorize(X_test[k,:,:], limits=limit)
            normalize_x_test = stats.zscore(winsorized_x_test, axis=None)
            X_test = normalize_x_test.T
            y_test = clf.decision_function(X_test)
            for j in range(20):
                start = (j)*6;
                stop  = (j+1)*6;
                stimulus_sequence = session.stimuli[start:stop]
                new_stimulus_sequence = stimulus_sequence - 1
                aa = scores[new_stimulus_sequence]
                aa = aa[:,None]
                bb = y_test[start:stop]
                cc = np.add(aa, bb)
                np.put(scores, new_stimulus_sequence, cc)
                ind = np.argmax(scores) + 1
                if ind == session.target:
                    n_correct[j] = n_correct[j] + 1
        n_correct_final[s,:] = n_correct
        mean_column_session = n_correct_final.mean(axis=0)
        acc = mean_column_session/3
    accuracy[str(k)] = np.mean(acc)*100
    
@app.post('/api/electrodes/cross_validation/')
async def analisis_cross_validation(cross: CrossValidation):

    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    solver = cross.solver
    tol = float(cross.tol)
    shrinkage = float(cross.shrinkage)
    cross = {
        "solver":solver,
        "tol":tol, 
        "shrinkage":shrinkage
    }
    accuracy : dict = {}
    await asyncio.gather(get_val_electrodes(cross, 0, accuracy)
                        , get_val_electrodes(cross, 1, accuracy)
                        , get_val_electrodes(cross, 2, accuracy)
                        , get_val_electrodes(cross, 3, accuracy)
                        , get_val_electrodes(cross, 4, accuracy)
                        , get_val_electrodes(cross, 5, accuracy)
                        , get_val_electrodes(cross, 6, accuracy)
                        , get_val_electrodes(cross, 7, accuracy)
                        , get_val_electrodes(cross, 8, accuracy)
                        , get_val_electrodes(cross, 9, accuracy)
                        , get_val_electrodes(cross, 10, accuracy)
                        , get_val_electrodes(cross, 11, accuracy)
                        , get_val_electrodes(cross, 12, accuracy)
                        , get_val_electrodes(cross, 13, accuracy)
                        , get_val_electrodes(cross, 14, accuracy)
                        , get_val_electrodes(cross, 15, accuracy)
                        , get_val_electrodes(cross, 16, accuracy)
                        , get_val_electrodes(cross, 17, accuracy)
                        , get_val_electrodes(cross, 18, accuracy)
                        , get_val_electrodes(cross, 19, accuracy)
                        , get_val_electrodes(cross, 20, accuracy)
                        , get_val_electrodes(cross, 21, accuracy)
                        , get_val_electrodes(cross, 22, accuracy)
                        , get_val_electrodes(cross, 23, accuracy)
                        , get_val_electrodes(cross, 24, accuracy)
                        , get_val_electrodes(cross, 25, accuracy)
                        , get_val_electrodes(cross, 26, accuracy)
                        , get_val_electrodes(cross, 27, accuracy)
                        , get_val_electrodes(cross, 28, accuracy)
                        , get_val_electrodes(cross, 29, accuracy)
                        , get_val_electrodes(cross, 30, accuracy)
                        , get_val_electrodes(cross, 31, accuracy))

    print('accuracy',accuracy)
    headers = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json"
    }
    data = { 
    "records": [
    {
    "fields": {
        "Date": date_time, 
        "E1":  accuracy["0"],
        "E2":  accuracy["1"],
        "E3":  accuracy["2"],
        "E4":  accuracy["3"],
        "E5":  accuracy["4"],
        "E6":  accuracy["5"],
        "E7":  accuracy["6"],
        "E8":  accuracy["7"],
        "E9":  accuracy["8"],
        "E10": accuracy["9"],
        "E11": accuracy["10"],
        "E12": accuracy["11"],
        "E13": accuracy["12"],
        "E14": accuracy["13"],
        "E15": accuracy["14"],
        "E16": accuracy["15"],
        "E17": accuracy["16"],
        "E18": accuracy["17"],
        "E19": accuracy["18"],
        "E20": accuracy["19"],
        "E21": accuracy["20"],
        "E22": accuracy["21"],
        "E23": accuracy["22"],
        "E24": accuracy["23"],
        "E25": accuracy["24"],
        "E26": accuracy["25"],
        "E27": accuracy["26"],
        "E28": accuracy["27"],
        "E29": accuracy["28"],
        "E30": accuracy["29"],
        "E31": accuracy["30"],
        "E32": accuracy["31"],
        "METODO": "CROSS-VALIDATION"
    }
    }
    ]
    }
  
    # r = requests.post(endpoint, json=data, headers=headers)
    data = accuracy
    return data

#########################PORCENTAJE##################################
@app.post('/api/electrodes/porcentage/')
async def analisis_cross_validation(porcentage: Porcentage):
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    count = 2
    runs = 6
    folder_data = 'sessions'
    entries = os.listdir(folder_data)
    se = ['s1.mat', 's2.mat', 's3.mat', 's4.mat']
    n_correct_final = np.zeros((4, 20))
    accuracy = []
    limit = 0.05
    solver = porcentage.solver
    tol = float(porcentage.tol)
    shrinkage = float(porcentage.shrinkage)
    test_size = float(porcentage.test_size)

    # 34 electrodos para hallar la exactitud
    for k in range(32):  
        
        # Cuatro Archivos para analizar
        X = np.empty((32, 32, 0))
        y = np.empty((0))
        
        X_train = np.empty((32, 32, 0))
        y_train = np.empty((0))
        
        X_test = np.empty((32, 32, 0))
        y_test = np.empty((0))
        
        # Resultados para cada sesion (6 estructuras)
        scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        n_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        for entry in se:
            file_selected = folder_data + '/' + entry
            data_session = loadmat(file_selected, squeeze_me=True, struct_as_record=False)
            
            # print(file_selected)
            
            for i in range(6):
                session = data_session['runs'][i]
                X = np.concatenate((X, session.x), axis=2)
                y = np.concatenate((y, session.y), axis=0)
                
        
        # Eliminacion de ruido y normalización
        winsorized_x = winsorize(X[k,:,:], limits=limit)
        normalize_x = stats.zscore(winsorized_x, axis=None)
        X = normalize_x.T
        
        # División del dataset en entrenamiento y testeo    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
                
        # # Entrenamiento del modelo LDA
        if solver == 'svd':
            clf = LinearDiscriminantAnalysis(solver=solver, tol=tol, store_covariance=False)
        else:
            clf = LinearDiscriminantAnalysis(solver=solver, tol=tol, store_covariance=False, shrinkage=shrinkage)

        clf.fit(X_train, y_train)
        
        session_test = se[1]
        
        file_selected = folder_data + '/' + session_test
            
            
        data_session = loadmat(file_selected, squeeze_me=True, struct_as_record=False)
        for i in range(6):
            session = data_session['runs'][i]
            X_test = session.x
            
            winsorized_x_test = winsorize(X_test[k,:,:], limits=limit)
            normalize_x_test = stats.zscore(winsorized_x_test, axis=None)
            X_test = normalize_x_test.T
            y_test = clf.decision_function(X_test)
            
            
            for j in range(20):
                start = (j)*6;
                stop  = (j+1)*6;
                stimulus_sequence = session.stimuli[start:stop]
                new_stimulus_sequence = stimulus_sequence - 1
                # print('new_stimulus_sequence', new_stimulus_sequence)
                aa = scores[new_stimulus_sequence]
                aa = aa[:,None]
                bb = y_test[start:stop]
                cc = np.add(aa, bb)
                np.put(scores, new_stimulus_sequence, cc)
                # print('scores', scores)
                ind = np.argmax(scores) + 1
                # print('ind', ind)
                # print(session.target, ind)
                if ind == session.target:
                    n_correct[j] = n_correct[j] + 1
                    # print('n_correct', n_correct)
            # print('n_correct', n_correct)

        
        n_correct_final[0,:] = n_correct
        
        mean_column_session = n_correct_final.mean(axis=0)
        
        acc = mean_column_session
        
        accuracy.append(np.mean(acc)*100)
    accuracy = {
        "0":  accuracy[0],
        "1":  accuracy[1],
        "2":  accuracy[2],
        "3":  accuracy[3],
        "4":  accuracy[4],
        "5":  accuracy[5],
        "6":  accuracy[6],
        "7":  accuracy[7],
        "8":  accuracy[8],
        "9":  accuracy[9],
        "10": accuracy[10],
        "11": accuracy[11],
        "12": accuracy[12],
        "13": accuracy[13],
        "14": accuracy[14],
        "15": accuracy[15],
        "16": accuracy[16],
        "17": accuracy[17],
        "18": accuracy[18],
        "19": accuracy[19],
        "20": accuracy[20],
        "21": accuracy[21],
        "22": accuracy[22],
        "23": accuracy[23],
        "24": accuracy[24],
        "25": accuracy[25],
        "26": accuracy[26],
        "27": accuracy[27],
        "28": accuracy[28],
        "29": accuracy[29],
        "30": accuracy[30],
        "31": accuracy[31],
    }

    headers = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json"
    }
    data = { 
    "records": [
    {
    "fields": {
        "Date": date_time, 
        "E1":  accuracy["0"],
        "E2":  accuracy["1"],
        "E3":  accuracy["2"],
        "E4":  accuracy["3"],
        "E5":  accuracy["4"],
        "E6":  accuracy["5"],
        "E7":  accuracy["6"],
        "E8":  accuracy["7"],
        "E9":  accuracy["8"],
        "E10": accuracy["9"],
        "E11": accuracy["10"],
        "E12": accuracy["11"],
        "E13": accuracy["12"],
        "E14": accuracy["13"],
        "E15": accuracy["14"],
        "E16": accuracy["15"],
        "E17": accuracy["16"],
        "E18": accuracy["17"],
        "E19": accuracy["18"],
        "E20": accuracy["19"],
        "E21": accuracy["20"],
        "E22": accuracy["21"],
        "E23": accuracy["22"],
        "E24": accuracy["23"],
        "E25": accuracy["24"],
        "E26": accuracy["25"],
        "E27": accuracy["26"],
        "E28": accuracy["27"],
        "E29": accuracy["28"],
        "E30": accuracy["29"],
        "E31": accuracy["30"],
        "E32": accuracy["31"],
        "METODO": "PORCENTAJE"
    }
    }
    ]
    }  
    r = requests.post(endpoint, json=data, headers=headers)
    data = accuracy
    return data