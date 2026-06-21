"""
Vercel Serverless Function Handler for Flask
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Create a WSGI callable for Vercel
def handler(environ, start_response):
    """WSGI handler for Vercel"""
    return app(environ, start_response)
