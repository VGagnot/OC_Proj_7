from flask import Flask, render_template


app = Flask(__name__)

app.config.from_object('config')
# To get one variable, tape app.config['MY_VARIABLE']

@app.route('/dashboard/')
def dashboard():
	return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)