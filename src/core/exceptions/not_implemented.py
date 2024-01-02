class NotImplementedException(Exception):
    def __init__(self, message: str):
        full_message = f'Not implemented error: {message}'
        super().__init__(full_message)
