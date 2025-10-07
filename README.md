# GameJamExternalScript
Game Jam 2025 External Script
Recommened to run in Python virtual environment. 
Requires MediaPipe (open source CV algorithm).
Instructions below.

Demo Video: https://youtu.be/W-xC9dKVPEU

If you are intersted in contributing to the project (let's make an awsome open source game!) feel free to pull from the Repo and do as you wish! Attribution is appreciated, but not nessecary. Game can be played on itch.io (https://gunderwood.itch.io/marios-mayhem-2) wihtout CV (not that fun). Email gunderwood@vt.edu if you want to contribute to the game (music, SFX, art, programming, gameplay, etc). 

Preparing the script for gameplay:
Requirments: Python 3.11 (MediaPipe does not currently have support for Python 3.12+)
1) If you don't already have it, downlonad Python 3.11.
2) Pull the Python script and store in a known location on your folder.
3) Create a Python virutal environemnt
   - On Windows: Use the Command Line or Terminal in VS Code (or your code editor) and type: python -m venv pose_env
   - Note: You are wlecome to chang the name of the environment. If you have multiple python verisons install, you may need to specify the verison of python when creating your environment using python3.11
   - Activate the virutal environment (the commands are different depending on the python version you have installed)
4) In the Command Line / Terminal: pip install mediapipe
5) In the Command Line / Terminal: pip install opencv-python
6) All dependencies should be installed now.

You should now be able to successfully run the Python script. In order to play the game with the CV controls you will also need to download the Godot game and run it locally as there is an issue with the server connection running it on web. The itch.io page has been updated.
