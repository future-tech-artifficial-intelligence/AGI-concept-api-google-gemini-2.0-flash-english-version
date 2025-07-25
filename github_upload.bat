```batch
@echo off
echo Commands to execute in the terminal to transfer your project to GitHub
echo Copy and paste these commands one by one into your terminal

echo.
echo Initializing the Git repository...
git init

echo.
echo Adding ALL files to Git tracking...
git add -A

echo.
echo Checking files that will be committed...
git status

echo.
echo Creating the initial commit...
git commit -m "AGI-concept-api-google-gemini-2.0-flash-english-version.git"

echo.
echo Configuring the remote repository...
git remote add origin https://github.com/future-tech-artifficial-intelligence/AGI-concept-api-google-gemini-2.0-flash-english-version.git

echo.
echo Pushing code to GitHub...
git push -u origin main

echo.
echo If the push fails, try with the force option:
echo git push -u -f origin main

echo.
echo Note: If your main branch is called "master" instead of "main", use:
echo git push -u origin master

echo.
echo Done! Your code has been transferred to GitHub.
pause
