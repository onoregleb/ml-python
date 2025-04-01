class MLModel:
    def __init__(self, id: str, name: str, cost: float, description: str = ""):
        self.id = id
        self.name = name
        self.cost = cost
        self.description = description

    def get_id(self):
        return self.id

    def get_name(self):
        return self.name

    def get_cost(self):
        return self.cost

    def get_description(self):
        return self.description