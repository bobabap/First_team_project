from flask import Blueprint
from flask import render_template
from flask import jsonify
from flask import request, redirect, url_for
from werkzeug.utils import secure_filename
from pydub import AudioSegment

import sys,os
import pandas as pd
import csv
import json

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import flask_app.database as database



# 경로확인
print(os.getcwd())

# 블루프린트 생성
bp = Blueprint("user", __name__, url_prefix='/')

# 메인페이지
@bp.route('/', methods=["GET", "POST"])
def main():
  return render_template('main.html')



# 음성파일 확장자 확인
WAV_EXTENSIONS = set(['wav'])
def wav_file(filename):
    return '.' in filename and \
    filename.rsplit('.', 1)[1] in WAV_EXTENSIONS


M4A_EXTENSIONS = set(['m4a'])
def m4a_file(filename):
    return '.' in filename and \
    filename.rsplit('.', 1)[1] in M4A_EXTENSIONS



@bp.route("/apply", methods=["POST", "GET"])
def apply():

  if request.method == 'POST':
    # 기침 이외의 데이터 수집
    id = 1
    age = request.form["age"]
    gender = request.form["gender"]
    respiratory_condition = request.form["respiratory_condition"]
    fever_or_muscle_pain = request.form["fever_or_muscle_pain"]
    print(age, gender,respiratory_condition,fever_or_muscle_pain)

    # 기침 데이터 수집
    file = request.files['file']

    # 파일형식 확인 후 저장
    if file and wav_file(file.filename):
      filename = secure_filename("new_cough.wav")
      file.save(os.path.join("flask_app/Data", filename))

    elif file and m4a_file(file.filename):
      track = AudioSegment.from_file(file,  format= 'm4a')
      filename = secure_filename("new_cough.wav")
      filename = file.save(os.path.join("flask_app/Data", filename))
      file_handle = track.export(filename, format='wav')

 
    # csv파일 저장
    database.save(id,age,gender,respiratory_condition,fever_or_muscle_pain)

    # 모델링 실행
    import flask_app.baseline as baseline

    test_result = pd.read_csv("flask_app/Data/new_result.csv")['covid19'][0]
    if test_result==1:
      test_result="코로나 양성"
    elif test_result==0:
      test_result="코로나 음성"

  # 결과 게시
  return render_template("apply.html", result = test_result)
  #return redirect(url_for("main.html"))  


@bp.route('/retest', methods=["GET", "POST"])
def retest():
  return render_template("retest.html")
