class User:
    """
    Класс, представляющий пользователя системы.
    
    Этот класс хранит информацию о пользователе, такую как идентификатор, имя,
    логин и пароль. Также предоставляет методы для проверки пароля.
    """
    def __init__(self, id: str, firstname: str, username: str, password: str):
        """
        Инициализирует новый экземпляр пользователя.
        
        Args:
            id (str): Уникальный идентификатор пользователя
            firstname (str): Имя пользователя
            username (str): Логин пользователя (уникальный)
            password (str): Пароль пользователя
        """
        self.id = id
        self.firstname = firstname
        self.username = username
        self.password = password

    def get_id(self):
        """
        Возвращает идентификатор пользователя.
        
        Returns:
            str: Идентификатор пользователя
        """
        return self.id

    def get_firstname(self):
        """
        Возвращает имя пользователя.
        
        Returns:
            str: Имя пользователя
        """
        return self.firstname

    def get_username(self):
        """
        Возвращает логин пользователя.
        
        Returns:
            str: Логин пользователя
        """
        return self.username

    def get_password(self):
        """
        Возвращает пароль пользователя.
        
        Returns:
            str: Пароль пользователя
        
        Note:
            Этот метод должен использоваться с осторожностью, так как он возвращает пароль в открытом виде.
        """
        return self.password

    def verify_password(self, password: str):
        """
        Проверяет, совпадает ли указанный пароль с паролем пользователя.
        
        Args:
            password (str): Пароль для проверки
            
        Returns:
            bool: True если пароль совпадает, иначе False
        
        Note:
            В настоящее время пароли хранятся в открытом виде, что не безопасно.
            В реальном приложении следует использовать хеширование паролей.
        """
        return self.password == password