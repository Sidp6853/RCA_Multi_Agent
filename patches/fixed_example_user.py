class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def get_info(self):
        return f"Name: {self.name}, Email: {self.email}"

    def update_email(self, new_email):
        self.email = new_email
        return f"Email updated to {self.email}"