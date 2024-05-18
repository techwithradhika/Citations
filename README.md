#  Property Management API

This project fetches data from an API and identifies the sources for each response.
It uses the Sentence Transformers model to match response texts with their respective sources. 
The identified sources, called citations, are then displayed in a user-friendly web interface built with Flask.

# Prerequisites

- Visual Studio Code.
- Python and pip installed.

## Installation

To run this project locally, follow these steps:

1. Clone the repository: git clone https://github.com/techwithradhika/Citations.git
2. In your terminal Navigate to the current project directory: cd Citations
3. Install virtual environment (recommended): pip install virtualenv
4. Create a virtual environment: python -m virtualenv venv
5. Activate the virtual environment: .\venv\Scripts\activate
6. Install the dependencies: pip install requests torch flask sentence-transformers
7. Run the Flask application: python main.py
8. Open your web browser and navigate to: http://127.0.0.1:5000/
9. View the citations: The citations will be displayed in a list format. Each citation includes the ID and link.
