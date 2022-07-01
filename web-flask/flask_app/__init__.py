import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from flask import Flask
import flask_app.user_route as user_routes
from flask import render_template
import csv


# 웹서비스
def create_app():
  app = Flask(__name__)
  app.register_blueprint(user_routes.bp)
  return app



if __name__ == "__main__":

  app = create_app()
  # 한글 깨짐 방지
  app.config["JSON_AS_ASCII"] = False
  
  

  app.run()

