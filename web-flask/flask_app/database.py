import pandas as pd


def save(id,age,gender,respiratory_condition,fever_or_muscle_pain):
  idx = len(pd.read_csv("flask_app/Data/new_data.csv"))
  new_df = pd.DataFrame({"id":id, "age":age, "gender":gender, "respiratory_condition":respiratory_condition, "fever_or_muscle_pain":fever_or_muscle_pain}, index=[idx])
  new_df.to_csv("flask_app/Data/new_data.csv", index=False)
  return None


