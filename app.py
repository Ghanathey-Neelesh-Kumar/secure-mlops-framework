from flask import Flask, request, render_template, jsonify
import logging
from cryptography.fernet import Fernet
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

class SecureMLOpsPipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.allowed_roles = ['admin', 'data_scientist', 'ml_engineer']
        self.access_attempts = []
        self.thresholds = {
            'anomaly': 0.05  # 5% anomaly threshold for detection
        }

    def configure_security_settings(self):
        self.pipeline.set_security(
            access_control="strict",
            encryption="AES-256",
            validation="comprehensive"
        )
        logging.info("Security settings configured.")

    def enforce_access_controls(self, user, role):
        if role not in self.allowed_roles:
            logging.warning(f"Unauthorized access attempt by user: {user}, role: {role}")
            raise PermissionError("Access Denied")
        logging.info(f"Access granted to user: {user}, role: {role}")
        self.log_access_attempt(user, role)

    def log_access_attempt(self, user, role):
        attempt = {'user': user, 'role': role}
        self.access_attempts.append(attempt)
        with open('access_log.txt', 'a') as log_file:
            log_file.write(f"User: {user}, Role: {role}, Attempt: {'Success' if role in self.allowed_roles else 'Failed'}\n")
        logging.info(f"Access attempt logged for user: {user}, role: {role}")
        self.detect_intrusion(attempt)

    def detect_intrusion(self, attempt):
        # Basic intrusion detection logic
        suspicious_roles = ['unauthorized_role', 'guest']
        if attempt['role'] in suspicious_roles:
            logging.error(f"Intrusion detected: {attempt}")
            self.raise_alert(attempt)

    def raise_alert(self, attempt):
        # Raise an alert for the intrusion attempt
        alert_message = f"ALERT: Intrusion detected for user: {attempt['user']}, role: {attempt['role']}"
        logging.critical(alert_message)
        with open('alerts.txt', 'a') as alert_file:
            alert_file.write(alert_message + "\n")

    def encrypt_data(self, data):
        key = Fernet.generate_key()
        cipher_suite = Fernet(key)
        encrypted_data = cipher_suite.encrypt(data.encode())
        logging.info("Data encrypted.")
        return encrypted_data, key

    def decrypt_data(self, encrypted_data, key):
        cipher_suite = Fernet(key)
        decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
        logging.info("Data decrypted.")
        return decrypted_data

    def validate_model(self, model, validation_data, threshold=0.8):
        accuracy = model.score(validation_data['X'], validation_data['y'])
        logging.info(f"Model validation performed. Accuracy: {accuracy}")
        if accuracy < threshold:
            logging.error("Model validation failed.")
            raise ValueError("Model validation failed")
        logging.info("Model validation passed.")
        self.detect_anomalies(model, validation_data)

    def detect_anomalies(self, model, validation_data):
        # Simple anomaly detection using prediction errors
        predictions = model.predict(validation_data['X'])
        errors = np.abs(predictions - validation_data['y'])
        mean_error = np.mean(errors)
        if mean_error > self.thresholds['anomaly']:
            logging.warning(f"Anomaly detected in model predictions. Mean error: {mean_error}")
            self.raise_alert({'user': 'model', 'role': 'anomaly_detected'})

# Mock pipeline class
class MockPipeline:
    def set_security(self, access_control, encryption, validation):
        self.access_control = access_control
        self.encryption = encryption
        self.validation = validation

    def get_security(self):
        return self.access_control, self.encryption, self.validation

# Load dataset and train model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Initialize pipeline
pipeline = MockPipeline()
secure_pipeline = SecureMLOpsPipeline(pipeline)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/configure_security', methods=['POST'])
def configure_security():
    secure_pipeline.configure_security_settings()
    return jsonify({'status': 'Security settings configured'})

@app.route('/enforce_access', methods=['POST'])
def enforce_access():
    user = request.form['user']
    role = request.form['role']
    try:
        secure_pipeline.enforce_access_controls(user, role)
        return jsonify({'status': 'Access granted'})
    except PermissionError:
        return jsonify({'status': 'Access denied'})

@app.route('/encrypt_data', methods=['POST'])
def encrypt():
    data = request.form['data']
    encrypted_data, key = secure_pipeline.encrypt_data(data)
    return jsonify({'encrypted_data': encrypted_data.decode(), 'key': key.decode()})

@app.route('/decrypt_data', methods=['POST'])
def decrypt():
    encrypted_data = request.form['encrypted_data']
    key = request.form['key']
    decrypted_data = secure_pipeline.decrypt_data(encrypted_data.encode(), key.encode())
    return jsonify({'decrypted_data': decrypted_data})

@app.route('/validate_model', methods=['POST'])
def validate():
    validation_data = {'X': X_test, 'y': y_test}
    try:
        secure_pipeline.validate_model(model, validation_data)
        return jsonify({'status': 'Model validation passed'})
    except ValueError:
        return jsonify({'status': 'Model validation failed'})

if __name__ == '__main__':
    app.run(debug=True)
