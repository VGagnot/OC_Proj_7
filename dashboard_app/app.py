import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import joblib
from lightgbm import LGBMClassifier
import shap
import plotly.graph_objs as go

lgbm2 = joblib.load(open("classification_credit.sav", 'rb'))
testy = pd.read_csv("testy.csv")
testX = pd.read_csv("extraitX.csv")
testX = testX.set_index(testX.columns[0])

explainerModel = shap.TreeExplainer(lgbm2)
shap_values = explainerModel.shap_values(testX)

import random
#i = 0
#i = random.randrange(0, len(testy)-1)
i = random.randrange(0, 9)

pred = lgbm2.predict_proba(testX.iloc[i].to_numpy().reshape(1, -1))[0][0]

app = dash.Dash()



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

top_10_contrib_idx = np.argsort(abs(shap_values[0][i]))[-10:]

lim_x = max(abs(shap_values[0][i]))*1.1

liste_contribs = []
val_contribs = []
col = []

for j in top_10_contrib_idx:
  liste_contribs.append(testX.columns[j])
  val_contribs.append(shap_values[0][i][j])
  if shap_values[0][i][j] > 0:
        col.append('red')
  else:
        col.append('green')


fig2, ax = plt.subplots(figsize=(12,12))
fig2 = plt.barh(y=liste_contribs, width=val_contribs, color = col)
ax.set_title('Principaux contributeurs')
ax.axes.xaxis.set_visible(False)
ax.set_xlim([-lim_x, lim_x])


#Layout:

app.layout = html.Div([
    dcc.Graph(id='jauge',
		figure = jauge
              )
#    ,dcc.Graph(id='fig2',
#              figure=fig2
#              )
    ,dcc.Graph(id='remb',
              figure=go.Figure(data=[Remb],
              layout=go.Layout())
              )
    ])


if __name__ == "__main__":
    app.run_server(debug=True)