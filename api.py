from google_auth_oauthlib.flow import InstalledAppFlow
import requests

# Constants
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
CLIENT_SECRET_FILE = 'client_secret_1074700202256-f68lbq4ke3qr92mvdsbus83lsntv5a0q.apps.googleusercontent.com.json'
API_URL = "https://generativelanguage.googleapis.com/v1beta2/models/gemini-1.5-flash:generateText"

# Function to get OAuth 2.0 credentials
def get_oauth_credentials():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)

    try:
        # Attempt to run the local server for OAuth flow
        credentials = flow.run_local_server(port=8080)  # Explicitly specify a port
    except Exception as e:
        print(f"Failed to run local server: {e}")
        print("Please go to this URL to authorize:")
        auth_url, _ = flow.authorization_url(access_type='offline')
        print(auth_url)

        # Prompt the user to enter the authorization code manually
        code = input("Enter the authorization code: ")
        credentials = flow.fetch_token(code=code)

    return credentials

# Function to test API authentication
def test_api_authentication(credentials):
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json"
    }

    # Prepare the request data
    data = {
        "prompt": "Test the authentication with a basic request.",
        "max_tokens": 5
    }

    try:
        # Send a POST request to the API
        response = requests.post(API_URL, headers=headers, json=data)
        response_json = response.json()
        if response.status_code == 200:
            print("Authentication successful! Response:", response_json)
        else:
            print("Error:", response_json)
    except Exception as e:
        print(f"API Request Error: {e}")

# Run the test
if __name__ == "__main__":
    credentials = get_oauth_credentials()
    test_api_authentication(credentials)
