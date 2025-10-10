import os
import json
from pymongo import MongoClient
from datetime import datetime

class MongoDBUploader:
    def __init__(self, uri, db_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def upload_json_file(self, file_path, collection_name, metadata=None):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        document = {
            "filename": os.path.basename(file_path),
            "data": data,
            "uploaded_at": datetime.utcnow()
        }
        if metadata:
            document["metadata"] = metadata
        result = self.db[collection_name].insert_one(document)
        print(f"Uploaded {file_path} to collection '{collection_name}' with _id {result.inserted_id}")
        return result.inserted_id

    def upload_results_directory(self, results_dir, project_name, model_name, run_timestamp):
        files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        metadata = {
            "project_name": project_name,
            "model_name": model_name,
            "run_timestamp": run_timestamp
        }
        for file in files:
            file_path = os.path.join(results_dir, file)
            collection_name = file.replace('.json', '')
            self.upload_json_file(file_path, collection_name, metadata)
