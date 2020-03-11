from flask import Flask, request, jsonify, render_template
from flask_cors import cross_origin
import json

from model import Speak
from model import  find_similarword
app = Flask(__name__)

# model = Speak.SpeakDetect()
model = find_similarword.SpeakDetect()

@app.route("/", methods=["GET"])
def main():
    return render_template("main.html")

@app.route('/api/model', methods=["POST"])
@cross_origin()
def api_model():
    try:
        input_body = request.form.get('body','')

        result = {
            "code": 0,
            "data":{
                "detail": model.get_sentiment(input_body)
            }
        }
        return jsonify(result)
    except Exception as e:
        print(str(e))
        return jsonify({"code":-1, "data":str(e)})

if __name__ == '__main__':
    app.run(port=7007, host="0.0.0.0")

application = app