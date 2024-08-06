import logging
from cryptography.fernet import Fernet
import unittest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Configure security settings
secure_pipeline.configure_security_settings()

# Enforce access controls
try:
    secure_pipeline.enforce_access_controls('user1', 'data_scientist')
except PermissionError as e:
    print(e)

# Encrypt and decrypt data
data = "sensitive information"
encrypted_data, key = secure_pipeline.encrypt_data(data)
decrypted_data = secure_pipeline.decrypt_data(encrypted_data, key)
print(f"Original: {data}, Decrypted: {decrypted_data}")

# Validate model with real data
validation_data = {'X': X_test, 'y': y_test}
try:
    secure_pipeline.validate_model(model, validation_data)
except ValueError as e:
    print(e)

# Run unit tests and integration tests
class TestSecureMLOpsPipeline(unittest.TestCase):
    class MockPipeline:
        def set_security(self, access_control, encryption, validation):
            self.access_control = access_control
            self.encryption = encryption
            self.validation = validation

        def get_security(self):
            return self.access_control, self.encryption, self.validation

    def setUp(self):
        self.mock_pipeline = self.MockPipeline()
        self.pipeline = SecureMLOpsPipeline(self.mock_pipeline)

        # Load dataset and train models
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
        self.model_high_accuracy = RandomForestClassifier()
        self.model_high_accuracy.fit(X_train, y_train)
        
        # Create a low accuracy model
        self.model_low_accuracy = RandomForestClassifier(max_depth=1, n_estimators=1)
        self.model_low_accuracy.fit(X_train, y_train)
        
        self.validation_data = {'X': X_test, 'y': y_test}

    def test_configure_security_settings(self):
        self.pipeline.configure_security_settings()
        security_settings = self.mock_pipeline.get_security()
        self.assertEqual(security_settings, ("strict", "AES-256", "comprehensive"))

    def test_enforce_access_controls(self):
        self.pipeline.enforce_access_controls('user1', 'data_scientist')
        with self.assertRaises(PermissionError):
            self.pipeline.enforce_access_controls('user2', 'unauthorized_role')

    def test_encrypt_decrypt_data(self):
        data = "test data"
        encrypted_data, key = self.pipeline.encrypt_data(data)
        decrypted_data = self.pipeline.decrypt_data(encrypted_data, key)
        self.assertEqual(data, decrypted_data)

    def test_validate_model_high_accuracy(self):
        self.pipeline.validate_model(self.model_high_accuracy, self.validation_data)

    def test_validate_model_low_accuracy(self):
        with self.assertRaises(ValueError):
            self.pipeline.validate_model(self.model_low_accuracy, self.validation_data, threshold=0.9)

class IntegrationTestSecureMLOpsPipeline(unittest.TestCase):
    def setUp(self):
        class MockPipeline:
            def set_security(self, access_control, encryption, validation):
                self.access_control = access_control
                self.encryption = encryption
                self.validation = validation

            def get_security(self):
                return self.access_control, self.encryption, self.validation

        # Load dataset and train models
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
        self.model_high_accuracy = RandomForestClassifier()
        self.model_high_accuracy.fit(X_train, y_train)
        
        # Create a low accuracy model
        self.model_low_accuracy = RandomForestClassifier(max_depth=1, n_estimators=1)
        self.model_low_accuracy.fit(X_train, y_train)
        
        self.validation_data = {'X': X_test, 'y': y_test}

        self.pipeline = MockPipeline()
        self.secure_pipeline = SecureMLOpsPipeline(self.pipeline)

    def test_pipeline_integration_high_accuracy(self):
        # Configure security settings
        self.secure_pipeline.configure_security_settings()
        security_settings = self.pipeline.get_security()
        self.assertEqual(security_settings, ("strict", "AES-256", "comprehensive"))

        # Enforce access controls
        self.secure_pipeline.enforce_access_controls('user1', 'data_scientist')
        with self.assertRaises(PermissionError):
            self.secure_pipeline.enforce_access_controls('user2', 'unauthorized_role')

        # Encrypt and decrypt data
        data = "integration test data"
        encrypted_data, key = self.secure_pipeline.encrypt_data(data)
        decrypted_data = self.secure_pipeline.decrypt_data(encrypted_data, key)
        self.assertEqual(data, decrypted_data)

        # Validate model with high accuracy
        self.secure_pipeline.validate_model(self.model_high_accuracy, self.validation_data)

    def test_pipeline_integration_low_accuracy(self):
        # Configure security settings
        self.secure_pipeline.configure_security_settings()
        security_settings = self.pipeline.get_security()
        self.assertEqual(security_settings, ("strict", "AES-256", "comprehensive"))

        # Enforce access controls
        self.secure_pipeline.enforce_access_controls('user1', 'data_scientist')
        with self.assertRaises(PermissionError):
            self.secure_pipeline.enforce_access_controls('user2', 'unauthorized_role')

        # Encrypt and decrypt data
        data = "integration test data"
        encrypted_data, key = self.secure_pipeline.encrypt_data(data)
        decrypted_data = self.secure_pipeline.decrypt_data(encrypted_data, key)
        self.assertEqual(data, decrypted_data)

        # Validate model with low accuracy
        with self.assertRaises(ValueError):
            self.secure_pipeline.validate_model(self.model_low_accuracy, self.validation_data, threshold=0.9)

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
