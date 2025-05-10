class MLModel:
    """
    Класс, представляющий модель машинного обучения.
    
    Этот класс хранит метаданные модели, такие как идентификатор, название,
    стоимость использования и описание. Используется для представления
    доступных моделей в API и пользовательском интерфейсе.
    """
    def __init__(self, id: str, name: str, cost: float, description: str = ""):
        """
        Инициализирует новый экземпляр модели машинного обучения.
        
        Args:
            id (str): Уникальный идентификатор модели
            name (str): Название модели
            cost (float): Стоимость использования модели
            description (str, optional): Описание модели. По умолчанию пустая строка.
        """
        self.id = id
        self.name = name
        self.cost = cost
        self.description = description

    def get_id(self):
        """
        Возвращает идентификатор модели.
        
        Returns:
            str: Идентификатор модели
        """
        return self.id

    def get_name(self):
        """
        Возвращает название модели.
        
        Returns:
            str: Название модели
        """
        return self.name

    def get_cost(self):
        """
        Возвращает стоимость использования модели.
        
        Returns:
            float: Стоимость использования модели
        """
        return self.cost

    def get_description(self):
        """
        Возвращает описание модели.
        
        Returns:
            str: Описание модели
        """
        return self.description