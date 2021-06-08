from flask import Flask, jsonify
import joblib
from lightgbm import LGBMClassifier
import shap
import numpy as np
import pandas as pd

app = Flask(__name__)

lgbm2 = joblib.load(open("classification_credit.sav", 'rb'))
testy = pd.read_csv("echantillon_test_y.csv")
testX = pd.read_csv("echantillon_test_X.csv")
testX = testX.set_index(testX.columns[0])

explainerModel = shap.TreeExplainer(lgbm2)
shap_values = explainerModel.shap_values(testX)

@app.route('/fi_locales/')
#def shap_loc_val(i):
def shap_loc_val():
	i=0
	pred = lgbm2.predict_proba(testX.iloc[i].to_numpy().reshape(1, -1))[0][0]
	top_10_contrib_idx = np.argsort(abs(shap_values[0][i]))[-10:]
	liste_contribs = []
	val_contribs = []
	col = []
	col_glob = []
	contrib_top_10_glob = [
		shap_values[0][i][testX.columns.tolist().index('NEW_EXT_SOURCES_SUM_stdscl')],
		shap_values[0][i][testX.columns.tolist().index('EXT_SOURCE_2_stdscl')],
		shap_values[0][i][testX.columns.tolist().index('EXT_SOURCE_3_stdscl')],
		shap_values[0][i][testX.columns.tolist().index('NEW_SOURCES_PROD_stdscl')],
		shap_values[0][i][testX.columns.tolist().index('INSTAL_DPD_MEAN_stdscl')],
		shap_values[0][i][testX.columns.tolist().index('NEW_CREDIT_TO_GOODS_RATIO_stdscl')],
		shap_values[0][i][testX.columns.tolist().index('NEW_CREDIT_TO_ANNUITY_RATIO_stdscl')],
		shap_values[0][i][testX.columns.tolist().index('CODE_GENDER_stdscl')],
		shap_values[0][i][testX.columns.tolist().index('NEW_DOC_IND_KURT_stdscl')],
		shap_values[0][i][testX.columns.tolist().index('AMT_ANNUITY_stdscl')]
		]

	for j in top_10_contrib_idx:
		liste_contribs.append(testX.columns[j])
		val_contribs.append(shap_values[0][i][j])
		if shap_values[0][i][j] > 0:
			col.append('red')
		else:
			col.append('green')

	for j in contrib_top_10_glob:
		if j > 0:
			col_glob.append('red')
		else:
			col_glob.append('green')

	lim_x = [
		max(map(abs, val_contribs))*1.1,
		max(map(abs, contrib_top_10_glob))*1.1,
		max(map(abs, contrib_top_10_glob + val_contribs))*1.1
		]
		

	data = [pred, liste_contribs, val_contribs, col, lim_x]
	return jsonify({
		'pred':pred,
		'liste_top_10_contribs': liste_contribs,
		'val_top_10_contribs': val_contribs,
		'col': col,
		'col_glob': col_glob,
		'contrib_top_10_glob': contrib_top_10_glob,
		'lim_x': lim_x})


if __name__ == "__main__":
    app.run(debug=True)