from flask import Flask, request, jsonify, render_template
from flask_cors import cross_origin
import json

from model import Speak
app = Flask(__name__)

@app.route("/", methods=["GET"])
def main():
    return render_template("main.html")

@app.route('/api/model', methods=["POST"])
@cross_origin()
def model():
    try:
        input_body = request.form.get('body','')

        model =  Speak.SpeakDetect()
        result = {
            "code": 0,
            "data":{
                "detail": model.get_speak_content(input_body)
            }
        }
        return jsonify(result)
    except Exception as e:
        print(str(e.__traceback__))
        return jsonify({"code":-1, "data":str(e)})


if __name__ == '__main__':
    app.run()

application = app