from flask import Flask, render_template, request, jsonify
import util

app = Flask(__name__, template_folder='templates')

@app.route('/image', methods = ['GET', 'POST'])
def home():
    # image_data = request.form['image_data']
    # response = jsonify(util.classify_image(image_data))


    # response.headers.add('Access-Control-Allow-Origin','*')

    return render_template('app.html')

if __name__ == '__main__':
    print("Starting Python Flask Server for Football Stars Image Classification")
    # util.load_saved_artifacts() 
    app.run(debug=True)

