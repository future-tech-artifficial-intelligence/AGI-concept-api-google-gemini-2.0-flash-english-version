import os
import subprocess
import sys
import shutil
from datetime import datetime

def run_command(command, description):
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return False

def create_version_tag():
    """Creates a version tag based on the current date."""
    current_date = datetime.now()
    version_tag = f"v{current_date.year}.{current_date.month:02d}.{current_date.day:02d}"
    
    # Check if the tag already exists
    try:
        result = subprocess.run(f"git tag -l {version_tag}", shell=True, capture_output=True, text=True)
        if version_tag in result.stdout:
            # If the tag exists, add a suffix with the time
            version_tag = f"v{current_date.year}.{current_date.month:02d}.{current_date.day:02d}-{current_date.hour:02d}{current_date.minute:02d}"
    except:
        pass
    
    return version_tag

def main():
    project_path = os.path.dirname(os.path.abspath(__file__))
    repo_url = "https://github.com/future-tech-artifficial-intelligence/AGI-concept-api-google-gemini-2.0-flash-english-version.git"
    
    print(f"Deploying the project to folder: {project_path}")
    print(f"To GitHub repository: {repo_url}")
    
    os.chdir(project_path)
    
    # Clean up existing Git repository if necessary
    if os.path.exists(".git"):
        print("\nExisting Git repository detected. Cleaning up...")
        try:
            # Remove the .git folder for a fresh start
            import shutil
            shutil.rmtree(".git", ignore_errors=True)
            print("Old Git repository successfully deleted.")
        except Exception as e:
            print(f"Warning: Could not delete old Git repository: {e}")
            print("Attempting forced re-initialization...")
    
    # Initialize Git repository
    if not run_command("git init", "Initializing Git repository"):
        return
    
    # Add files to Git tracking
    if not run_command("git add --all", "Adding all files to Git tracking"):
        return
    
    # Create initial commit
    commit_message = "Full upload of the AGI-ASI-AI project open-source
    commit_result = subprocess.run(f'git commit -m "{commit_message}"', shell=True, capture_output=True, text=True)
    
    # Check if commit failed because there's nothing to commit
    if "nothing to commit" in commit_result.stderr or "nothing to commit" in commit_result.stdout:
        print("No changes detected, all files are already committed. Continuing deployment...")
    elif commit_result.returncode != 0:
        # If another error occurred
        print(f"Error during commit: {commit_result.stderr}")
        return
    
    # Create a version tag to enable releases
    version_tag = create_version_tag()
    if not run_command(f'git tag -a {version_tag} -m "Release {version_tag} - Added Termux/Android compatibility"', 
                      f"Creating version tag {version_tag}"):
        print("Warning: Could not create tag, but continuing...")
    
    # Configure remote repository
    # Check if remote origin already exists
    try:
        subprocess.run("git remote get-url origin", shell=True, check=True, capture_output=True, text=True)
        print("Remote 'origin' already exists. Deleting for reconfiguration...")
        run_command("git remote remove origin", "Deleting existing remote")
    except subprocess.CalledProcessError:
        pass  # Remote does not exist, continue normally
    
    if not run_command(f'git remote add origin {repo_url}', "Configuring remote repository"):
        return
    
    # Determine default branch
    print("\nDetecting default branch...")
    default_branch = "main"  # Most new repositories use "main" as the default branch
    
    # Push code to GitHub
    if not run_command(f"git push -u origin {default_branch}", f"Pushing code to branch {default_branch}"):
        print("\nAttempting to push to 'master' branch instead...")
        if not run_command("git push -u origin master", "Pushing code to master branch"):
            print("\nDeployment failed. Check your GitHub credentials and repository settings.")
            return
        default_branch = "master"
    
    # Push tags to GitHub
    if not run_command("git push origin --tags", "Pushing tags to GitHub"):
        print("Warning: Could not push tags, but code has been transferred.")
    
    print(f"\nDone! Your code has been successfully transferred to GitHub.")
    print(f"Repository URL: {repo_url}")
    print(f"Tag created: {version_tag}")
    print(f"\nTo create a release on GitHub:")
    print(f"1. Go to {repo_url}/releases")
    print(f"2. Click on 'Create a new release'")
    print(f"3. Select tag '{version_tag}'")
    print(f"4. Add a title and description for your release")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")
