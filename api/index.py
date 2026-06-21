import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Export the app for Vercel
def handler(environ, start_response):
    return app(environ, start_response)
