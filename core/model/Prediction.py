from typing import Optional, Dict, Any
from datetime import datetime


class Prediction:
    """
    Класс, представляющий предсказание модели машинного обучения.
    
    Этот класс хранит информацию о предсказаниях, включая входные данные,
    результаты, статус и информацию о времени создания/обновления.
    Предоставляет методы для управления статусом и результатами предсказания.
    """
    def __init__(self, id: str, model_id: str, user_id: str, input_data: Dict[str, Any],
                 output_data: Optional[Dict[str, Any]] = None, status: str = "pending"):
        """
        Инициализирует новый экземпляр предсказания.
        
        Args:
            id (str): Уникальный идентификатор предсказания
            model_id (str): Идентификатор используемой модели
            user_id (str): Идентификатор пользователя, создавшего предсказание
            input_data (Dict[str, Any]): Входные данные для предсказания
            output_data (Optional[Dict[str, Any]], optional): Результаты предсказания. По умолчанию None.
            status (str, optional): Статус предсказания. По умолчанию "pending".
        """
        self.id = id
        self.model_id = model_id
        self.user_id = user_id
        self.input_data = input_data
        self.output_data = output_data
        self.status = status
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def get_id(self):
        """
        Возвращает идентификатор предсказания.
        
        Returns:
            str: Идентификатор предсказания
        """
        return self.id

    def get_model_id(self):
        """
        Возвращает идентификатор модели, использованной для предсказания.
        
        Returns:
            str: Идентификатор модели
        """
        return self.model_id

    def get_user_id(self):
        """
        Возвращает идентификатор пользователя, создавшего предсказание.
        
        Returns:
            str: Идентификатор пользователя
        """
        return self.user_id

    def get_input_data(self):
        """
        Возвращает входные данные предсказания.
        
        Returns:
            Dict[str, Any]: Входные данные предсказания
        """
        return self.input_data

    def get_output_data(self):
        """
        Возвращает результаты предсказания.
        
        Returns:
            Optional[Dict[str, Any]]: Результаты предсказания или None, если предсказание еще не завершено
        """
        return self.output_data

    def get_status(self):
        """
        Возвращает текущий статус предсказания.
        
        Returns:
            str: Статус предсказания ('pending', 'completed', 'failed')
        """
        return self.status

    def set_output_data(self, output_data: Dict[str, Any]):
        """
        Устанавливает результаты предсказания и обновляет статус на 'completed'.
        
        Также обновляет время последнего обновления предсказания.
        
        Args:
            output_data (Dict[str, Any]): Результаты предсказания
        """
        self.output_data = output_data
        self.updated_at = datetime.now()
        self.status = "completed"

    def set_failed(self):
        """
        Устанавливает статус предсказания в 'failed'.
        
        Используется в случае ошибки при обработке предсказания.
        Также обновляет время последнего обновления предсказания.
        """
        self.status = "failed"
        self.updated_at = datetime.now()