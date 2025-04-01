class Account:
    def __init__(self, id: str, user_id: str, balance: float):
        self.id = id
        self.user_id = user_id
        self.balance = balance

    def get_id(self):
        return self.id

    def get_user_id(self):
        return self.user_id

    def get_balance(self):
        return self.balance

    def add_balance(self, amount: float):
        self.balance += amount
        return self.balance

    def subtract_balance(self, amount: float):
        if self.balance >= amount:
            self.balance -= amount
            return True
        return False