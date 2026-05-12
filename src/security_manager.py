# src/security_manager.py

import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

class SecurityManager:
    """
    Clase encargada de la gestión de llaves y criptografía simétrica.
    Garantiza que los vectores faciales y metadatos sean ilegibles fuera
    de la aplicación.
    """

    def __init__(self, key_path='data/security/secret.key'):
        self.key_path = key_path
        self.key = self._initialize_key()
        self.cipher_suite = Fernet(self.key)

    def _initialize_key(self):
        """
        Carga la llave existente o genera una nueva si es la primera ejecución.
        La llave se almacena en la ruta de seguridad definida en el diseño.
        """
        if os.path.exists(self.key_path):
            with open(self.key_path, 'rb') as key_file:
                return key_file.read()
        else:
            # Generación de llave AES-256 (Fernet)
            key = Fernet.generate_key()
            # Asegurar que el directorio existe antes de escribir
            os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
            with open(self.key_path, 'wb') as key_file:
                key_file.write(key)
            return key

    def encrypt_data(self, data: bytes) -> bytes:
        """
        Cifra datos binarios. Utilizado para proteger los 'embeddings' 
        antes de la persistencia[cite: 1].
        """
        return self.cipher_suite.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Descifra datos binarios. Permite recuperar los vectores para
        la comparación en el motor de inferencia[cite: 1].
        """
        return self.cipher_suite.decrypt(encrypted_data)
        
    def _derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """Deriva una llave de 32 bytes a partir de una contraseña usando PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def export_key_backup(self, password: str, backup_path: str):
        """
        Cifra la llave maestra actual y la guarda en una ubicación externa.
        El archivo resultante contiene: [16 bytes de SALT] + [LLAVE_MAESTRA_CIFRADA]
        """
        salt = os.urandom(16)
        backup_key = self._derive_key_from_password(password, salt)
        backup_cipher = Fernet(backup_key)
        
        encrypted_master_key = backup_cipher.encrypt(self.key)
        
        with open(backup_path, 'wb') as f:
            f.write(salt + encrypted_master_key)
        return True

    def import_key_backup(self, password: str, backup_path: str):
        """
        Lee un backup, lo descifra con la contraseña y restaura la secret.key original.
        """
        if not os.path.exists(backup_path):
            return False
            
        with open(backup_path, 'rb') as f:
            data = f.read()
            
        salt = data[:16]
        encrypted_master_key = data[16:]
        
        try:
            backup_key = self._derive_key_from_password(password, salt)
            backup_cipher = Fernet(backup_key)
            restored_key = backup_cipher.decrypt(encrypted_master_key)
            
            # Sobrescribir la llave actual en disco y en memoria
            os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
            with open(self.key_path, 'wb') as f:
                f.write(restored_key)
            
            self.key = restored_key
            self.cipher_suite = Fernet(self.key)
            return True
        except Exception:
            return False # Contraseña incorrecta o archivo corrupto