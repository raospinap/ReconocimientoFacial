# src/security_manager.py
"""Núcleo criptográfico del sistema. 
Gestiona el ciclo de vida de la llave maestra simétrica (Fernet/AES-256).
Garantiza que ningún vector biométrico o metadato personal se almacene en texto plano y proporciona mecanismos seguros de respaldo y restauración mediante derivación de claves (PBKDF2). 
Alineado estrictamente con el principio de Privacidad desde el Diseño y la Ley 1581 de 2012"""

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
        """Configura ruta de secret.key, invoca inicialización segura y monta la suite de cifrado Fernet.
        RNF-05, RNF-06, RNF-07 """
        self.key_path = key_path
        self.key = self._initialize_key()
        self.cipher_suite = Fernet(self.key)

    def _initialize_key(self):
        """Carga llave existente o genera nueva clave AES-256 en primera ejecución, asegurando permisos de directorio.
        RNF-05, RNF-07  """
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
        """Cifra blobs binarios (vectores, auditoría) usando Fernet, garantizando confidencialidad y autenticidad.
        RF-01, RF-04, RNF-05   """
        return self.cipher_suite.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Descifra datos protegidos para su procesamiento en memoria, validando firma interna de Fernet.
        RF-01, RNF-05
        """
        return self.cipher_suite.decrypt(encrypted_data)
        
    def _derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """Deriva clave de 32 bytes desde contraseña de usuario mediante PBKDF2-HMAC-SHA256 (100k iteraciones).
        RNF-05, RNF-09 (Seguridad)"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def export_key_backup(self, password: str, backup_path: str):
        """Genera salt aleatorio, deriva clave temporal, cifra la master key y concatena [SALT + CIPHERTEXT] en .vault.
        RNF-05, RNF-06, RNF-07 """
        salt = os.urandom(16)
        backup_key = self._derive_key_from_password(password, salt)
        backup_cipher = Fernet(backup_key)
        
        encrypted_master_key = backup_cipher.encrypt(self.key)
        
        with open(backup_path, 'wb') as f:
            f.write(salt + encrypted_master_key)
        return True

    def import_key_backup(self, password: str, backup_path: str):
        """Lee backup, extrae salt, deriva clave, descifra master key y actualiza estado en runtime/disco.
        RNF-05, RNF-06, RNF-07 """
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