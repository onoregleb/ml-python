class User:
    def __init__(self, id, firstname, username, password):
        """Init a user"""
        self.id = id
        self.firstname = firstname
        self.username = username
        self.password = password

    def get_id(self):
        return self.id
    def get_username(self):
        return self.username
    def get_password(self):
        return self.password
