#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import json
from typing import Dict, List, Set
import re

def create_directory_structure():
    """Create the new FastAPI backend directory structure."""
    directories = [
        'backend/app',
        'backend/app/api/v1/endpoints',
        'backend/app/core',
        'backend/app/crud',
        'backend/app/db',
        'backend/app/models',
        'backend/app/schemas',
        'backend/app/services',
        'backend/app/utils',
        'backend/tests',
        'backend/alembic',
        'backend/alembic/versions'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, '__init__.py'), 'w') as f:
            f.write('"""Module initialization."""\n')

def backup_existing_backend():
    """Backup existing backend code to archive."""
    if os.path.exists('backend'):
        timestamp = Path('backend').stat().st_mtime
        backup_dir = f'archive/backend_backup_{int(timestamp)}'
        shutil.copytree('backend', backup_dir)
        print(f"Backed up existing backend to: {backup_dir}")

def convert_js_routes_to_fastapi(js_file: str) -> str:
    """Convert Express routes to FastAPI endpoints."""
    with open(js_file, 'r') as f:
        content = f.read()
    
    # Extract route information
    routes = []
    route_pattern = r'router\.(get|post|put|delete)\([\'"]([^\'"]+)[\'"]'
    
    for match in re.finditer(route_pattern, content):
        method, path = match.groups()
        routes.append({
            'method': method.upper(),
            'path': path,
            'original_file': js_file
        })
    
    # Generate FastAPI endpoint template
    template = []
    for route in routes:
        endpoint_name = route['path'].strip('/').replace('/', '_')
        template.append(f"""
@router.{route['method'].lower()}("{route['path']}", response_model=schemas.{endpoint_name.title()}Response)
async def {endpoint_name}(
    *,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    \"\"\"
    {route['method']} {route['path']}
    Original file: {route['original_file']}
    \"\"\"
    # TODO: Implement endpoint logic
    raise NotImplementedError()
""")
    
    return '\n'.join(template)

def convert_mongoose_to_sqlalchemy(js_file: str) -> str:
    """Convert Mongoose schemas to SQLAlchemy models."""
    with open(js_file, 'r') as f:
        content = f.read()
    
    # Extract schema information
    schema_pattern = r'new Schema\({\s*([^}]+)\s*}\)'
    field_pattern = r'(\w+):\s*{\s*type:\s*(\w+)[^}]*}'
    
    schemas = []
    for schema_match in re.finditer(schema_pattern, content):
        fields = {}
        schema_content = schema_match.group(1)
        
        for field_match in re.finditer(field_pattern, schema_content):
            field_name, field_type = field_match.groups()
            fields[field_name] = field_type
        
        schemas.append(fields)
    
    # Generate SQLAlchemy model
    type_mapping = {
        'String': 'String',
        'Number': 'Float',
        'Date': 'DateTime',
        'Boolean': 'Boolean',
        'ObjectId': 'Integer'
    }
    
    template = []
    for i, schema in enumerate(schemas):
        class_name = Path(js_file).stem.title()
        template.append(f"""
class {class_name}(Base):
    \"\"\"SQLAlchemy model for {class_name}\"\"\"
    __tablename__ = "{class_name.lower()}s"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
""")
        
        for field_name, field_type in schema.items():
            sql_type = type_mapping.get(field_type, 'String')
            template.append(f"    {field_name} = Column({sql_type})")
    
    return '\n'.join(template)

def create_fastapi_app():
    """Create the main FastAPI application file."""
    content = '''"""Main FastAPI application module."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.api import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)
'''
    
    with open('backend/app/main.py', 'w') as f:
        f.write(content)

def create_config():
    """Create configuration files."""
    settings_content = '''"""Settings module."""
from typing import List, Union
from pydantic import AnyHttpUrl, BaseSettings, validator

class Settings(BaseSettings):
    """Application settings."""
    PROJECT_NAME: str = "Investment Recommendation System"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """Validate CORS origins."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    SQLALCHEMY_DATABASE_URI: str = None

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: str, values: dict) -> str:
        """Assemble database URI."""
        if isinstance(v, str):
            return v
        return f"postgresql://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_SERVER')}/{values.get('POSTGRES_DB')}"

    class Config:
        """Pydantic config."""
        case_sensitive = True
        env_file = ".env"

settings = Settings()
'''
    
    with open('backend/app/core/config.py', 'w') as f:
        f.write(settings_content)

def create_dependencies():
    """Create dependency injection module."""
    content = '''"""Dependencies module."""
from typing import Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from pydantic import ValidationError
from sqlalchemy.orm import Session
from app import crud, models, schemas
from app.core import security
from app.core.config import settings
from app.db.session import SessionLocal

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/login/access-token"
)

def get_db() -> Generator:
    """Get database session."""
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

async def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(reusable_oauth2)
) -> models.User:
    """Get current user from token."""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        token_data = schemas.TokenPayload(**payload)
    except (jwt.JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    user = crud.user.get(db, id=token_data.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
'''
    
    with open('backend/app/api/deps.py', 'w') as f:
        f.write(content)

def create_database_session():
    """Create database session module."""
    content = '''"""Database session module."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_engine(settings.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
'''
    
    with open('backend/app/db/session.py', 'w') as f:
        f.write(content)

def create_requirements():
    """Create requirements.txt with necessary packages."""
    requirements = '''# API
fastapi>=0.68.0,<0.69.0
uvicorn>=0.15.0,<0.16.0
python-multipart>=0.0.5,<0.0.6
email-validator>=1.1.3,<1.2.0

# Auth
python-jose[cryptography]>=3.3.0,<3.4.0
passlib[bcrypt]>=1.7.4,<1.8.0

# Database
SQLAlchemy>=1.4.23,<1.5.0
alembic>=1.7.1,<1.8.0
psycopg2-binary>=2.9.1,<2.10.0
greenlet>=1.1.1,<1.2.0

# ML/AI
numpy>=1.21.2,<1.22.0
pandas>=1.3.3,<1.4.0
scikit-learn>=0.24.2,<0.25.0
tensorflow>=2.6.0,<2.7.0
torch>=1.9.0,<1.10.0
transformers>=4.10.0,<4.11.0

# Utils
python-dotenv>=0.19.0,<0.20.0
requests>=2.26.0,<2.27.0
Jinja2>=3.0.1,<3.1.0
aiofiles>=0.7.0,<0.8.0
'''
    
    with open('backend/requirements.txt', 'w') as f:
        f.write(requirements)

def main():
    """Main migration function."""
    print("Starting backend migration...")
    
    # Backup existing backend
    backup_existing_backend()
    
    # Create new directory structure
    print("Creating new directory structure...")
    create_directory_structure()
    
    # Create core files
    print("Creating core FastAPI files...")
    create_fastapi_app()
    create_config()
    create_dependencies()
    create_database_session()
    create_requirements()
    
    # Convert Express routes to FastAPI endpoints
    print("Converting Express routes to FastAPI endpoints...")
    for root, _, files in os.walk('backend'):
        for file in files:
            if file.endswith('.js') and 'routes' in root:
                js_file = os.path.join(root, file)
                endpoint_file = f"backend/app/api/v1/endpoints/{Path(file).stem}.py"
                with open(endpoint_file, 'w') as f:
                    f.write(convert_js_routes_to_fastapi(js_file))
    
    # Convert Mongoose schemas to SQLAlchemy models
    print("Converting Mongoose schemas to SQLAlchemy models...")
    for root, _, files in os.walk('backend'):
        for file in files:
            if file.endswith('.js') and 'models' in root:
                js_file = os.path.join(root, file)
                model_file = f"backend/app/models/{Path(file).stem}.py"
                with open(model_file, 'w') as f:
                    f.write(convert_mongoose_to_sqlalchemy(js_file))
    
    print("\nMigration complete!")
    print("Next steps:")
    print("1. Review and adjust generated endpoints in backend/app/api/v1/endpoints/")
    print("2. Review and adjust generated models in backend/app/models/")
    print("3. Set up environment variables in .env file")
    print("4. Run database migrations using alembic")

if __name__ == "__main__":
    main() 