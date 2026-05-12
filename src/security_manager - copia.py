# src/security_manager.py

import os
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