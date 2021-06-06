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

	for j in top_10_contrib_idx:
		liste_contribs.append(testX.columns[j])
		val_contribs.append(shap_values[0][i][j])
		if shap_values[0][i][j] > 0:
			col.append('red')
		else:
			col.append('green')
	lim_x = max(abs(shap_values[0][i]))*1.1

	data = [pred, liste_contribs, val_contribs, col, lim_x]
	return jsonify({'data':data})


if __name__ == "__main__":
    app.run(debug=True)