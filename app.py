import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Carica i dati completi
path = "train_leggero.csv" if os.path.exists("train_leggero.csv") else "train.csv"
df = pd.read_csv(path)
if 'id' in df.columns: 
    df = df.drop(columns=['id'])
df = df.drop_duplicates().fillna(0)

# Pulizia nomi colonne
df.columns = [c.strip().lower() for c in df.columns]

# Codifica i testi in numeri
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

# Allena il modello globale sulle 13 colonne
target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target]

model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X, y)

# Salva il modello pronto in un file leggero
with open("modello_pronto.pkl", "wb") as f:
    pickle.dump(model, f)
print("Modello salvato con successo!")
