URI = ("mongodb+srv://YOOCHAN:200101@coivd19-cough-detection.xb8ze.mongodb.net/?retryWrites=true&w=majority")

from pymongo import MongoClient

client = MongoClient(URI)

print(client)

DATABASE = 'coivd19-cough-detection'

database = client[DATABASE]

COLLECTION = 'Collection1'

collection = database[COLLECTION]

print(collection)

collection.insert_one(document={"test":1})