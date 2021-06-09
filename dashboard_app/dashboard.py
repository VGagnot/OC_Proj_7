import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import requests
import pandas as pd
import numpy as np
import random

import joblib
from lightgbm import LGBMClassifier
import shap
import plotly.graph_objs as go

from flask import Flask

app = dash.Dash()



echantillon_test_X = pd.read_csv("echantillon_test_X.csv")
echantillon_test_X = echantillon_test_X.set_index(echantillon_test_X.columns[0])
echantillon_test_y = pd.read_csv("echantillon_test_y.csv")

# Pour le scatterplot: extrait d'individus du train pour population de référence
# Comprend classe de 0 à 4 selon la probabilité de défaut (echantillon_train_pred)

echantillon_train_pred = pd.read_csv("echantillon_train_pred.csv")
echantillon_train_pred = echantillon_train_pred.set_index(echantillon_train_pred.columns[0])
echantillon_train_X = pd.read_csv("echantillon_train_X.csv")
echantillon_train_X = echantillon_train_X.set_index(echantillon_train_X.columns[0])
feat_imp_glob = pd.read_csv("feat_imp_glob.csv")
feat_imp_glob['Feature2'] = feat_imp_glob['Feature'].copy()
feat_imp_glob = feat_imp_glob.set_index(feat_imp_glob['Feature2'])





#i = 0
#i = random.randrange(0, len(echantillon_test_y)-1)
i = random.randrange(0, len(echantillon_test_y))


data_requests = requests.get("http://127.0.0.1:5000/fi_locales/?individu="+str(i))
json_data = data_requests.json()
pred = json_data['pred']
liste_contribs_loc = json_data['liste_top_10_contribs']
val_contribs_loc = json_data['val_top_10_contribs']
col_contribs_loc = json_data['col']
lim_x = json_data['lim_x']



#Ma jauge:

seuil = 0.505
Echelle_seuil = [0, min(0.6*seuil,1), min(0.9*seuil,1), min(seuil,1),min(1.2*seuil,1), 1]

col = 'red'
if pred < seuil:
  col = 'green'
else:
  col = 'red'

jauge = go.Figure(go.Indicator(
    mode = "gauge+number",
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

#Contributions locales

contribs_loc = [go.Bar(
   x = val_contribs_loc,
   y = liste_contribs_loc,
   name = 'Local feature importance',
   marker_color = col_contribs_loc,
   orientation='h'
)]
ppaux_contribs_loc = go.Figure(data=contribs_loc)
ppaux_contribs_loc.update_layout(xaxis_range=[-lim_x[0],lim_x[0]])



#Contributions locales vs globales

top_10_contrib_glob_idx = np.argsort(abs(feat_imp_glob['Feature_importance']))[-10:]
top_10_contrib_glob = feat_imp_glob.iloc[top_10_contrib_glob_idx]

ppaux_contribs_glob = go.Figure()

ppaux_contribs_glob.add_trace(go.Bar(
   x = top_10_contrib_glob['Feature_importance'],
   y = top_10_contrib_glob['Feature'],
   name = 'Global feature importance',
   marker_color = 'gray',
   orientation='h'
))

ppaux_contribs_glob.add_trace(go.Bar(
   x = json_data['contrib_top_10_glob'],
   y = ['NEW_EXT_SOURCES_SUM_stdscl', 'EXT_SOURCE_2_stdscl', 'EXT_SOURCE_3_stdscl', 'NEW_SOURCES_PROD_stdscl', 'INSTAL_DPD_MEAN_stdscl', 'NEW_CREDIT_TO_GOODS_RATIO_stdscl', 'NEW_CREDIT_TO_ANNUITY_RATIO_stdscl', 'CODE_GENDER_stdscl', 'NEW_DOC_IND_KURT_stdscl', 'AMT_ANNUITY_stdscl'],
   name = 'Local feature importance',
   marker_color = json_data['col_glob'],
   orientation='h'
))

lim_x_loc_glob = max(lim_x[1], max(top_10_contrib_glob['Feature_importance'])*1,1)
ppaux_contribs_glob.update_layout(xaxis_range=[-lim_x_loc_glob, lim_x_loc_glob])









#Layout:

app.layout = html.Div(children=[
	html.H1(children='Dashboard credit'),

	html.Div(children='Bienvenue sur mon dashboard crédit.'),
	html.Div(["Choisissez un individu (0 à 999): ",
		dcc.Input(id='ID', 
			value='0', 
			type='number',
			min=0,
			max=999)]),
	html.Div(id='display_number_output'),

	dcc.Graph(id='jauge',
		figure = jauge
              )
	,dcc.Graph(id='ppaux_contribs_glob',
              figure=ppaux_contribs_glob
              )
	,dcc.Graph(id='ppaux_contribs_loc',
              figure=ppaux_contribs_loc
              )

	,dcc.Dropdown(id='dropdown_1', 
		multi=False, 
		options=[{'label': x, 'value': x} for x in sorted(echantillon_train_X.columns)],
		value="EXT_SOURCE_1_stdscl")
	,dcc.Dropdown(id='dropdown_2', 
		multi=False, 
		options=[{'label': x, 'value': x} for x in sorted(echantillon_train_X.columns)],
		value="EXT_SOURCE_2_stdscl")
	,dcc.Graph(id='scat',
              figure={}
		)
    ])


@app.callback(
	Output('display_number_output', 'children'),
	Input('ID', 'value')
	)
def afficher_candidat(ID):
	return u'Vous affichez le candidat "{}".'.format(ID)

@app.callback(
	Output(component_id='scat', component_property='figure'),
	Input(component_id='dropdown_1', component_property = 'value'),
	Input(component_id='dropdown_2', component_property = 'value')
	)
	#Scatter Plot
def update_scatter_plot(feat_x, feat_y):
	a = feat_x
	b = feat_y
	scat = go.Figure()
	scat.add_trace(go.Scatter(x=echantillon_train_X[echantillon_train_pred['Classe'] == 0][a],
                                y=echantillon_train_X[echantillon_train_pred['Classe'] == 0][b],
				mode='markers',
				marker_color='limegreen',
				name="Très bons candidats"
				))
	scat.add_trace(go.Scatter(x=echantillon_train_X[echantillon_train_pred['Classe'] == 1][a],
                                y=echantillon_train_X[echantillon_train_pred['Classe'] == 1][b],
				mode='markers',
				marker_color='lightgreen',
				name="Bons candidats"
				))
	scat.add_trace(go.Scatter(x=echantillon_train_X[echantillon_train_pred['Classe'] == 2][a],
                                y=echantillon_train_X[echantillon_train_pred['Classe'] == 2][b],
				mode='markers',
				marker_color='yellow',
				name="Candidats acceptables"
				))
	scat.add_trace(go.Scatter(x=echantillon_train_X[echantillon_train_pred['Classe'] == 3][a],
                                y=echantillon_train_X[echantillon_train_pred['Classe'] == 3][b],
				mode='markers',
				marker_color='orange',
				name="Candidats risqués"
				))
	scat.add_trace(go.Scatter(x=echantillon_train_X[echantillon_train_pred['Classe'] == 4][a],
                                y=echantillon_train_X[echantillon_train_pred['Classe'] == 4][b],
				mode='markers',
				marker_color='orangered',
				name="Candidats très risqués"
				))
	scat.add_trace(go.Scatter(x=[echantillon_test_X[a].iloc[i]],
                                y=[echantillon_test_X[b].iloc[i]],
				mode='markers',
				marker=dict(
            				color='black',
					size=12
				),
				name="Votre candidat"
				))
	scat.update_layout(
		autosize=False,
		width=1000,
		height=700)
	return scat



if __name__ == "__main__":
    app.run_server(debug=True)