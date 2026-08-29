import os
import certifi
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError
from datetime import datetime

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.connected = False
        self._connect()

    def _connect(self):
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            print("[WARN] MONGO_URI not found in environment variables. Database integration is disabled.")
            return

        try:
            # Use certifi to avoid SSL issues on some systems when connecting to Atlas
            self.client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
            
            # Verify connection
            self.client.admin.command('ping')
            
            # Use the default database specified in the URI, or fallback to 'medical_diagnostics'
            self.db = self.client.get_default_database(default='medical_diagnostics')
            self.collection = self.db['diagnoses']
            
            self.connected = True
            print(f"[OK] Successfully connected to MongoDB Atlas (Database: {self.db.name})")
            
        except (ConnectionFailure, ConfigurationError) as e:
            print(f"[ERROR] Could not connect to MongoDB: {e}")
            self.client = None
            self.connected = False
        except Exception as e:
            print(f"[ERROR] Unexpected MongoDB connection error: {e}")
            self.client = None
            self.connected = False

    def save_diagnosis(self, diagnosis_data):
        """
        Saves a diagnosis record to the database.
        Returns the inserted ID on success, or None on failure/if not connected.
        """
        if not self.connected:
            print("[WARN] Database not connected. Diagnosis not saved.")
            return None

        try:
            # Add a server-side timestamp
            diagnosis_data['created_at'] = datetime.utcnow()
            
            result = self.collection.insert_one(diagnosis_data)
            
            # Convert the ObjectId to a string so Flask's jsonify can serialize the dictionary
            diagnosis_data['_id'] = str(result.inserted_id)
            
            print(f"[OK] Diagnosis saved to MongoDB with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"[ERROR] Failed to save diagnosis to MongoDB: {e}")
            return None

    def get_recent_diagnoses(self, limit=10):
        """
        Retrieves the most recent diagnoses from the database.
        Returns a list of dictionaries.
        """
        if not self.connected:
            return []

        try:
            # Sort by created_at descending (newest first)
            cursor = self.collection.find({}, {'_id': 0}).sort('created_at', -1).limit(limit)
            return list(cursor)
        except Exception as e:
            print(f"[ERROR] Failed to retrieve diagnoses from MongoDB: {e}")
            return []

# Initialize a global instance
db = Database()
