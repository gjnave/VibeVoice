@echo off
setlocal ENABLEDELAYEDEXPANSION

:: ==== config ====
set REPO_URL=https://github.com/gjnave/VibeVoice.git

:: ==== start fresh ====
echo Cleaning old Git state...
rmdir /s /q .git 2>nul

:: ==== .gitignore ====
echo Creating .gitignore...
echo .venv/>> .gitignore
echo venv/>> .gitignore
echo __pycache__/>> .gitignore
echo .ipynb_checkpoints/>> .gitignore
echo .pytest_cache/>> .gitignore
echo *.log>> .gitignore
echo *.tmp>> .gitignore
echo .DS_Store>> .gitignore
echo .vscode/>> .gitignore
echo .idea/>> .gitignore
echo outputs/>> .gitignore
echo voices/*.wav>> .gitignore

:: ==== keep voices folder visible without audio ====
if not exist voices mkdir voices
type nul > voices\.gitkeep

:: ==== strip any old LFS wav rules from .gitattributes (safe if file missing) ====
if exist .gitattributes type .gitattributes | findstr /i /v ".wav" > .gitattributes.clean
if exist .gitattributes del .gitattributes
if exist .gitattributes.clean ren .gitattributes.clean .gitattributes

:: ==== reinit and push ====
echo Initializing new repo...
git init
git add .
git commit -m "Initial commit - clean start"

git branch -M main
git remote add origin %REPO_URL%

:: ensure upstream and overwrite remote to match local clean state
git push -u --force origin main

echo Done. Repo pushed to: %REPO_URL%
endlocal
