class User:
    def __init__(self, id: str, firstname: str, username: str, password: str):
        self.id = id
        self.firstname = firstname
        self.username = username
        self.password = password

    def get_id(self):
        return self.id

    def get_firstname(self):
        return self.firstname

    def get_username(self):
        return self.username

    def get_password(self):
        return self.password

    def verify_password(self, password: str):
        return self.password == password