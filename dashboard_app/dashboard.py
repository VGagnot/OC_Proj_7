import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib
from lightgbm import LGBMClassifier
import shap
import plotly.graph_objs as go

from flask import Flask

app = dash.Dash()

lgbm2 = joblib.load(open("classification_credit.sav", 'rb'))
testX = pd.read_csv("echantillon_test_X.csv")
testX = testX.set_index(echantillon_test_X.columns[0])

explainerModel = shap.TreeExplainer(lgbm2)
shap_values = explainerModel.shap_values(testX)

import random
#i = 0
#i = random.randrange(0, len(testy)-1)
i = random.randrange(0, 9)

pred = lgbm2.predict_proba(testX.iloc[i].to_numpy().reshape(1, -1))[0][0]



#Un graph très simple, pour test: les poids relatifs des remboursements et des défauts

Remb = testy.query("TARGET == 0")
Def = testy.query("TARGET == 1")
Remb = go.Bar(
    x=['Remboursement', 'Défaut'],
    y=[len(Remb), len(Def)],
    name='graph 1'
)


#Ma jauge:

seuil = 0.505
Echelle_seuil = [0, min(0.6*seuil,1), min(0.9*seuil,1), min(seuil,1),min(1.2*seuil,1), 1]

col = 'red'
if pred < seuil:
  col = 'green'
else:
  col = 'red'

jauge = go.Figure(go.Indicator(
    mode = "gauge",
    value = pred,
    domain = {'x': [0, 1], 'y': [0, 1]}
    ,title = {'text': "Score credit"}
    ,gauge = {'axis': {'range': [0, 1]}
              ,'bar': {'color': col}
              ,'steps': [
                  {'range': [0, Echelle_seuil[1]], 'color': 'limegreen'},
                  {'range': [Echelle_seuil[1], Echelle_seuil[2]], 'color': 'lightgreen'},
                  {'range': [Echelle_seuil[2], Echelle_seuil[3]], 'color': 'yellow'},
                  {'range': [Echelle_seuil[3], Echelle_seuil[4]], 'color': 'orange'},
                  {'range': [Echelle_seuil[4], 1], 'color': 'orangered'}]
             ,'threshold' : {'line': {'color': "black", 'width': 1}, 'thickness': 1, 'value': seuil}
              }
    ))


#Principaux contributeurs:


#fig2, ax = plt.subplots(figsize=(12,12))
#fig2 = plt.barh(y=liste_contribs, width=val_contribs, color = col)
#ax.set_title('Principaux contributeurs')
#ax.axes.xaxis.set_visible(False)
#ax.set_xlim([-lim_x, lim_x])

data_requests = requests.get("http://127.0.0.1:5000/fi_locales/")

#Layout:

app.layout = html.Div([
    dcc.Graph(id='jauge',
		figure = jauge
              )
#    ,dcc.Graph(id='fig2',
#              figure=[fig2,ax]
#              )
    ,dcc.Graph(id='remb',
              figure=go.Figure(data=[Remb],
              layout=go.Layout())
              )
    ])


if __name__ == "__main__":
    app.run_server(debug=True)