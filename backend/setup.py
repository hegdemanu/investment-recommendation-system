from setuptools import setup, find_packages

setup(
    name="investment_app",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'fastapi>=0.104.0',
        'uvicorn>=0.24.0',
        'sqlalchemy>=2.0.25',
        'python-jose[cryptography]>=3.3.0',
        'passlib[bcrypt]>=1.7.4',
        'python-multipart>=0.0.6',
        'pydantic>=2.5.0',
        'pydantic-settings>=2.1.0',
        'python-dotenv>=1.0.0',
    ],
) 