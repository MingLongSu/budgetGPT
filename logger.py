class Logger:
    def logging(self, text):
        print(f"[ LOGGING ]: { text }")
    
    def error(self, text):
        print(f"[ ERROR ]: { text }")

    def debug(self, text):
        print(f"[ DEBUG ]: { text }")

    def warning(self, text):
        print(f"[ WARNING ]: { text }")
        