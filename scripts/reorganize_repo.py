#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the new directory structure."""
    directories = [
        'backend/api',
        'backend/models',
        'backend/services',
        'frontend/components',
        'frontend/pages',
        'frontend/public',
        'trading_engine/models',
        'trading_engine/strategies',
        'trading_engine/backtesting',
        'data_pipeline/connectors',
        'data_pipeline/processors',
        'data_pipeline/storage',
        'docs/api',
        'docs/architecture',
        'docs/guides',
        'scripts/deployment',
        'scripts/setup',
        'scripts/maintenance',
        'archive'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / '.gitkeep').touch()

def move_files():
    """Move files to their new locations."""
    moves = [
        # Backend files
        ('app/', 'backend/'),
        ('server/', 'backend/'),
        
        # Frontend files
        ('client/', 'frontend/'),
        ('mern-dashboard/', 'frontend/'),
        
        # Trading engine files
        ('models/', 'trading_engine/models/'),
        ('run_investment_system.py', 'trading_engine/'),
        ('train_sentiment_models.py', 'trading_engine/'),
        
        # Data pipeline files
        ('data/', 'data_pipeline/'),
        
        # Documentation
        ('docs/', 'docs/'),
        ('CHANGELOG.md', 'docs/'),
        ('README.md', '.'),
        
        # Scripts
        ('scripts/', 'scripts/'),
        ('setup_modular_dashboard.py', 'scripts/setup/'),
        
        # Configuration files
        ('.env', '.'),
        ('.env.template', '.'),
        ('requirements.txt', '.'),
        ('docker-compose.yml', '.'),
        ('package.json', '.')
    ]
    
    for src, dst in moves:
        try:
            if os.path.exists(src):
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                else:
                    shutil.copytree(src, dst, dirs_exist_ok=True)
        except Exception as e:
            print(f"Error moving {src} to {dst}: {e}")

def cleanup_old_structure():
    """Remove old directories and files after successful migration."""
    to_remove = [
        'app',
        'server',
        'client',
        'mern-dashboard',
        'models',
        'data',
        'scripts/old',
        'project-Mern',
        '__pycache__',
        '.turbo',
        'node_modules'
    ]
    
    for item in to_remove:
        try:
            if os.path.isfile(item):
                os.remove(item)
            elif os.path.isdir(item):
                shutil.rmtree(item)
        except Exception as e:
            print(f"Error removing {item}: {e}")

def main():
    """Main function to reorganize the repository."""
    print("Starting repository reorganization...")
    
    # Create new directory structure
    print("Creating new directory structure...")
    create_directory_structure()
    
    # Move files to new locations
    print("Moving files to new locations...")
    move_files()
    
    # Cleanup old structure
    print("Cleaning up old structure...")
    cleanup_old_structure()
    
    print("Repository reorganization complete!")

if __name__ == "__main__":
    main() 