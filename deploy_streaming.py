import ftplib
import os

# Deployment script for LCL Sovereign Streaming
HOST = "ftp.launchcloudlabs.com"
USER = "command@launchcloudlabs.com"
PASS = "Nebulousi2c!" # Provided in memory/prompt
DIR  = "public_html/demo/streaming"

def deploy():
    print(f"Connecting to {HOST}...")
    try:
        ftp = ftplib.FTP(HOST)
        ftp.login(USER, PASS)
        
        # Ensure directory exists
        try:
            ftp.mkd(DIR)
            print(f"Created directory {DIR}")
        except:
            print(f"Directory {DIR} already exists or cannot be created.")

        ftp.cwd(DIR)

        # Upload files
        for filename in ["server_receiver.py", "index.html"]:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    ftp.storbinary(f"STOR {filename}", f)
                    print(f"Uploaded {filename}")
            else:
                print(f"Error: {filename} not found locally.")

        ftp.quit()
        print("Deployment Complete.")
    except Exception as e:
        print(f"FTP Error: {e}")

if __name__ == "__main__":
    deploy()
