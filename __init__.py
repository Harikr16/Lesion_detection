import os

from flask import Flask,request,render_template
from .inference import *

def create_app(test_config=None):
	# create and configure the app
	app = Flask(__name__, instance_relative_config=True)
	app.config.from_mapping(
		SECRET_KEY='dev',
		DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
	)

	if test_config is None:
		# load the instance config, if it exists, when not testing
		app.config.from_pyfile('config.py', silent=True)
	else:
		# load the test config if passed in
		app.config.from_mapping(test_config)

	# ensure the instance folder exists
	try:
		os.makedirs(app.instance_path)
	except OSError:
		pass

	# a simple page that says hello
	@app.route('/hello')
	def hello():
		return 'Hello, World!'
		
	@app.route('/yo')
	def yo():
		return 'yo,yo,yo !!!'

	@app.route('/html' ,methods=['GET','POST'])
	def hml():
		if request.method == 'GET':
			return render_template('index.html',value="hello")
		if request.method == 'POST':
			if 'file' not in request.files:
				print("File not uploaded")  
				print(request.files)  
				print(request)
				return
			
			print("File uploaded")
			file = request.files['file']
			import io    
			import base64
			superimposed,confidence,found = get_cancer_prediction(file.read())	
			imgByteArr = io.BytesIO()
			superimposed.save(imgByteArr, format='PNG')
			imgByteArr = imgByteArr.getvalue()
			data_uri = base64.b64encode(imgByteArr).decode('utf-8')

			if confidence<0.3:
				return render_template('expert.html')
			else: 
				return render_template('result.html',value={'img':data_uri,'found':found})
			

	return app