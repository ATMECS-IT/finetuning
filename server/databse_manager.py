import os
import json
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime
import logging
from typing import Dict, Optional, List

class DatabaseManager:    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.db = None
        self.enabled = config.get("database", {}).get("mongodb_enabled", "false").lower() == "true"     
        if self.enabled:
            self._connect()
            self.logger.info("DatabaseManager initialized and connected")
        else:
            self.logger.warning("DatabaseManager is disabled in configuration")
    
    def _connect(self):
        try:
            uri = self.config["database"]["mongodb_uri"]
            db_name = self.config["database"]["mongodb_db_name"]
            
            self.client = MongoClient(
                uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000
            )
            self.client.admin.command('ping')
            self.logger.info(f"Successfully connected to MongoDB at {uri}")

            self.db = self.client[db_name]
            self.logger.info(f"Using database: {db_name}")
            
            self._create_indexes()
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            self.enabled = False
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to MongoDB: {e}")
            self.enabled = False
    
    def _create_indexes(self):
        try:
            self.db.training_runs.create_index([
                ("project_name", 1),
                ("model_name", 1),
                ("timestamp", -1)
            ])
            
            self.db.step_metrics.create_index([("run_id", 1), ("step", 1)])
            self.db.step_metrics.create_index([("project_name", 1)])
            self.db.step_metrics.create_index([("model_name", 1)])
            self.db.step_metrics.create_index([("timestamp", 1)])
            self.db.step_metrics.create_index([("project_name", 1), ("model_name", 1), ("timestamp", -1)])
            
            self.db.epoch_metrics.create_index([("run_id", 1), ("epoch", 1)])
            self.db.epoch_metrics.create_index([("project_name", 1)])
            self.db.epoch_metrics.create_index([("model_name", 1)])
            self.db.epoch_metrics.create_index([("timestamp", 1)])
            self.db.epoch_metrics.create_index([("project_name", 1), ("model_name", 1), ("timestamp", -1)])
            
            self.db.final_metrics.create_index([("run_id", 1)])
            self.db.final_metrics.create_index([("project_name", 1)])
            self.db.final_metrics.create_index([("model_name", 1)])
            self.db.final_metrics.create_index([("timestamp", 1)])
            self.db.final_metrics.create_index([("project_name", 1), ("model_name", 1), ("timestamp", -1)])
            
            self.db.evalutor.create_index([("project_name", 1)])
            self.db.evalutor.create_index([("model_name", 1)])
            self.db.evalutor.create_index([("timestamp", -1)])
            self.db.evalutor.create_index([("project_name", 1), ("model_name", 1), ("timestamp", -1)])
            self.logger.info("Created MongoDB indexes")
        except Exception as e:
            self.logger.warning(f"Failed to create indexes: {e}")
    
    def create_training_run(self, run_metadata: Dict) -> Optional[str]:
        if not self.enabled:
            self.logger.warning("DatabaseManager is not enabled")
            return None
        
        try:
            run_metadata["timestamp"] = datetime.utcnow()
            run_metadata["status"] = "running"
            
            result = self.db.training_runs.insert_one(run_metadata)
            run_id = str(result.inserted_id)
            
            self.logger.info(f"Created training run with ID: {run_id}")
            return run_id
            
        except Exception as e:
            self.logger.error(f"Failed to create training run: {e}")
            return None
    
    def save_evaluation_comparator(self, eval_data: dict):
        if not self.enabled:
            self.logger.warning("DatabaseManager is not enabled")
            return
        try:
            eval_data["saved_at"] = datetime.utcnow()
            self.db.evalutor.insert_one(eval_data)
            self.logger.info("Saved evaluation_comparator document to evalutor collection")
        except Exception as e:
            self.logger.error(f"Failed to save evaluation_comparator: {e}")


    def update_training_run(self, run_id: str, update_data: Dict):
        if not self.enabled or not run_id:
            return
        
        try:
            from bson.objectid import ObjectId
            
            update_data["last_updated"] = datetime.utcnow()
            
            self.db.training_runs.update_one(
                {"_id": ObjectId(run_id)},
                {"$set": update_data}
            )
            
            self.logger.debug(f"Updated training run {run_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update training run: {e}")
    
    def save_step_metrics(self, run_id: str, step_metrics: List[Dict], project_name: str, model_name: str, timestamp: str):
        if not self.enabled or not run_id or not step_metrics:
            return
        try:
            for metric in step_metrics:
                metric["run_id"] = run_id
                metric["project_name"] = project_name
                metric["model_name"] = model_name
                metric["timestamp"] = timestamp
                metric["saved_at"] = datetime.utcnow()
            self.db.step_metrics.insert_many(step_metrics)
            self.logger.info(f"Saved {len(step_metrics)} step metrics for run {run_id}")
        except Exception as e:
            self.logger.error(f"Failed to save step metrics: {e}")    

    def save_epoch_metrics(self, run_id: str, epoch_metrics: List[Dict], project_name: str, model_name: str, timestamp: str):
        if not self.enabled or not run_id or not epoch_metrics:
            return
        try:
            for metric in epoch_metrics:
                metric["run_id"] = run_id
                metric["project_name"] = project_name
                metric["model_name"] = model_name
                metric["timestamp"] = timestamp
                metric["saved_at"] = datetime.utcnow()
            self.db.epoch_metrics.insert_many(epoch_metrics)
            self.logger.info(f"Saved {len(epoch_metrics)} epoch metrics for run {run_id}")
        except Exception as e:
            self.logger.error(f"Failed to save epoch metrics: {e}")
    
    def save_final_metrics(self, run_id: str, final_metrics: Dict, project_name: str, model_name: str, timestamp: str):
        if not self.enabled or not run_id:
            return
        try:
            final_metrics["run_id"] = run_id
            final_metrics["project_name"] = project_name
            final_metrics["model_name"] = model_name
            final_metrics["timestamp"] = timestamp
            final_metrics["saved_at"] = datetime.utcnow()
            self.db.final_metrics.insert_one(final_metrics)
            self.update_training_run(run_id, {
                "status": "completed",
                "completed_at": datetime.utcnow(),
                "final_metrics_summary": {
                    "total_training_time": final_metrics.get("training_summary", {}).get("total_training_time"),
                    "total_epochs": final_metrics.get("training_summary", {}).get("total_epochs"),
                    "final_train_loss": final_metrics.get("training_summary", {}).get("final_train_loss"),
                    "best_val_loss": final_metrics.get("training_summary", {}).get("best_eval_loss")
                }
            })
            self.logger.info(f"Saved final metrics for run {run_id}")
        except Exception as e:
            self.logger.error(f"Failed to save final metrics: {e}")
    
    def save_model_comparison(self, run_id: str, comparisons: List[Dict]):
        if not self.enabled or not run_id or not comparisons:
            return
        
        try:
            for comparison in comparisons:
                if isinstance(comparison, str):
                    self.logger.warning(f"Comparison entry is a string, converting to dict. Value: {comparison}")
                    comparison = {"value": comparison}
                else:
                    comparison["run_id"] = run_id
                    comparison["saved_at"] = datetime.utcnow()
            
            self.db.model_comparisons.insert_many(comparisons)
            self.logger.info(f"Saved {len(comparisons)} model comparisons for run {run_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model comparisons: {e}")
    
    def get_training_runs(self, project_name: Optional[str] = None, 
                         model_name: Optional[str] = None,
                         limit: int = 10) -> List[Dict]:
        if not self.enabled:
            return []
        
        try:
            query = {}
            if project_name:
                query["project_name"] = project_name
            if model_name:
                query["model_name"] = model_name
            
            runs = list(self.db.training_runs
                       .find(query)
                       .sort("timestamp", -1)
                       .limit(limit))
            
            for run in runs:
                run["_id"] = str(run["_id"])
            
            return runs
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve training runs: {e}")
            return []
    
    def get_run_metrics(self, run_id: str) -> Dict:
        if not self.enabled or not run_id:
            return {}
        
        try:
            from bson.objectid import ObjectId
            
            run = self.db.training_runs.find_one({"_id": ObjectId(run_id)})
            if run:
                run["_id"] = str(run["_id"])
            
            step_metrics = list(self.db.step_metrics.find({"run_id": run_id}))
            epoch_metrics = list(self.db.epoch_metrics.find({"run_id": run_id}))
            final_metrics = self.db.final_metrics.find_one({"run_id": run_id})
            comparisons = list(self.db.model_comparisons.find({"run_id": run_id}))
            
            return {
                "run_info": run,
                "step_metrics": step_metrics,
                "epoch_metrics": epoch_metrics,
                "final_metrics": final_metrics,
                "model_comparisons": comparisons
            }
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve run metrics: {e}")
            return {}
    
    def close(self):
        if self.client:
            self.client.close()
            self.logger.info("Closed MongoDB connection")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
