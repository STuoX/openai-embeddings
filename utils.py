import openai

#OpenAI API Key location
ENV_PATH = ".env"

def get_openai_api_key(env_path=ENV_PATH):
    # read and parse file
    with open(env_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("OPENAI_API_KEY"):
            openai_api_key = line.split("=")[1].strip()
    print(f'Read OpenAI API authentication key from {env_path} file')
    return openai_api_key

def openai_auth(openai_api_key:str=get_openai_api_key()):
    # authenticate with OpenAI
    openai.api_key = openai_api_key
    print(f'Set OpenAI authentication key')
    return