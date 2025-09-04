import os
from dotenv import load_dotenv

class Config:
    
    def __init__(self):
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')

        if not os.path.exists(dotenv_path):
            print(f"Warning: .env file not found at {dotenv_path}, using system environment variables.")
        
        load_dotenv(dotenv_path)

        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        if not self.OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")
            

if __name__ == '__main__':
    config = Config()
    print(config.CHANNEL_SECRET)
