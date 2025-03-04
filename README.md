# DL-AI_Final_assignment-4-


## Setting Up the Python 3.10 Virtual Environment

Follow the steps below to create and activate a virtual environment and install the required Python dependencies.

### Install Python 3.10: 
#### Installing Python 3.10 on macOS (With Homebrew Option)

#### MacOS: Install Homebrew (If Needed)
If you don't already have Homebrew installed, you can install it by running the following command in your terminal:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Download Python 3.10 on MacOS using Homebrew:**
```bash
brew install python@3.10
```

#### On Windows

1. Download the Python 3.10.6 installer from the [official Python website]([https://www.python.org/downloads/release/python-31016/]).
2. Run the installer and follow the installation steps. Make sure to check the option "Add Python 3.10 to PATH" during installation.


### 1. Create a Virtual Environment

Run the following command to create a virtual environment using Python 3.10:

```bash
python3.10 -m venv venv
```
This will create a folder named venv in the project root.

### 2. Activate the Virtual Environment
Activate the virtual environment with the following command:

#### On Linux/macOS:
```bash
source venv/bin/activate
```
#### On Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies
With the virtual environment active, install the project's dependencies from the requirements.txt file using:

```bash
pip install -r requirements.txt
```
