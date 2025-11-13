from google_auth_oauthlib.flow import InstalledAppFlow

CLIENT_CONFIG = {
    "installed": {
        "client_id": "906666849556-77j0au4g5robghnqnta8k7psh84idhuv.apps.googleusercontent.com",
        "client_secret": "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",
        "redirect_uris": ["http://localhost"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token"
    }
}
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def main():
    flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, SCOPES)
    creds = flow.run_local_server(port=0)
    print("Access token:", creds.token)
    print("Refresh token:", creds.refresh_token)

if __name__ == "__main__":
    main()
