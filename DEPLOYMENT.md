# Deployment and GitHub checklist

## Local run
Use `run_windows.bat` on Windows, or follow README.md.

## GitHub
1. Create a new repository named `CropWise-Ultimate`.
2. Do not upload `.venv`, `cropwise.db`, model artifacts, secrets, or dataset files unless you have checked their size and redistribution license.
3. From the project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial CropWise Ultimate project"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/CropWise-Ultimate.git
   git push -u origin main
   ```
   Replace the URL with your own repository URL.

## Streamlit Community Cloud
1. Push the project to GitHub.
2. Sign in to Streamlit Community Cloud and deploy the repository.
3. Set the main file path to `app.py` (repository root is this folder).
4. Install dependencies from `requirements.txt`.
5. Upload the dataset in the app's Train & Compare Models page.

Note: Community Cloud's local filesystem may be ephemeral; SQLite history and trained artifacts may not persist reliably across restarts. For persistent production deployment, use a managed database/object store. This project is a demo, not a production agricultural advisory service.
