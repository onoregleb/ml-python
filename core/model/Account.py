class Account:
    """
    Класс, представляющий аккаунт пользователя.
    
    Этот класс хранит информацию об аккаунте пользователя, включая баланс.
    Предоставляет методы для работы с балансом, такие как пополнение и списание средств.
    """
    def __init__(self, id: str, user_id: str, balance: float):
        """
        Инициализирует новый экземпляр аккаунта.
        
        Args:
            id (str): Уникальный идентификатор аккаунта
            user_id (str): Идентификатор пользователя, которому принадлежит аккаунт
            balance (float): Текущий баланс аккаунта
        """
        self.id = id
        self.user_id = user_id
        self.balance = balance

    def get_id(self):
        """
        Возвращает идентификатор аккаунта.
        
        Returns:
            str: Идентификатор аккаунта
        """
        return self.id

    def get_user_id(self):
        """
        Возвращает идентификатор пользователя, которому принадлежит аккаунт.
        
        Returns:
            str: Идентификатор пользователя
        """
        return self.user_id

    def get_balance(self):
        """
        Возвращает текущий баланс аккаунта.
        
        Returns:
            float: Текущий баланс
        """
        return self.balance

    def add_balance(self, amount: float):
        """
        Увеличивает баланс аккаунта на указанную сумму.
        
        Args:
            amount (float): Сумма, на которую увеличивается баланс
            
        Returns:
            float: Новый баланс после пополнения
        """
        self.balance += amount
        return self.balance

    def subtract_balance(self, amount: float):
        """
        Уменьшает баланс аккаунта на указанную сумму, если на балансе достаточно средств.
        
        Args:
            amount (float): Сумма, на которую уменьшается баланс
            
        Returns:
            bool: True если операция успешна (достаточно средств), иначе False
        """
        if self.balance >= amount:
            self.balance -= amount
            return True
        return False