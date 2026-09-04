class AppException(Exception):
    def __init__(self, message, error=None):
        super().__init__(f"{message}: {error}" if error else message)
