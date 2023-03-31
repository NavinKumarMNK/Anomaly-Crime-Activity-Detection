"@Author: NavinKumarMNK"
# Custom Exceptions 
class VideoTooShort(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
    
    def __str__(self):
        return self.message

class VideoNotOpened(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
    
    def __str__(self):
        return self.message
