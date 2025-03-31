#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def move_files():
    """Move remaining files to their correct locations."""
    moves = [
        # Move frontend related files
        ('client/*', 'frontend/'),
        ('apps/web/*', 'frontend/'),
        
        # Move backend related files
        ('apps/api/*', 'backend/'),
        ('apps/server/*', 'backend/'),
        ('run.py', 'backend/'),
        
        # Move trading engine files
        ('run_investment_system.py', 'trading_engine/'),
        ('train_sentiment_models.py', 'trading_engine/models/'),
        
        # Move data pipeline files
        ('notebooks/*', 'data_pipeline/notebooks/'),
        ('data/*', 'data_pipeline/data/'),
        
        # Move documentation
        ('CHANGELOG.md', 'docs/'),
        ('REPOSITORY-STRUCTURE.md', 'docs/'),
        ('PLACEHOLDER-MANAGEMENT.md', 'docs/'),
        ('PLACEHOLDERS.md', 'docs/'),
        
        # Move configuration files
        ('config/*', 'backend/config/'),
        
        # Move test files
        ('tests/*', 'backend/tests/'),
        
        # Move logs
        ('logs/*', 'data_pipeline/logs/'),
        ('*.log', 'data_pipeline/logs/'),
        
        # Move results and reports
        ('results/*', 'data_pipeline/results/'),
        ('reports/*', 'data_pipeline/reports/')
    ]
    
    for src, dst in moves:
        try:
            if '*' in src:
                # Handle glob patterns
                for file in Path().glob(src):
                    if file.is_file():
                        os.makedirs(os.path.dirname(dst + file.name), exist_ok=True)
                        shutil.copy2(file, dst + file.name)
                    else:
                        shutil.copytree(file, dst + file.name, dirs_exist_ok=True)
            else:
                if os.path.exists(src):
                    if os.path.isfile(src):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(src, dst)
                    else:
                        shutil.copytree(src, dst, dirs_exist_ok=True)
        except Exception as e:
            print(f"Error moving {src} to {dst}: {e}")

def cleanup_directories():
    """Remove unnecessary directories and files."""
    to_remove = [
        'apps',
        'client',
        'data',
        'notebooks',
        'logs',
        'results',
        'reports',
        '__pycache__',
        '.turbo',
        'node_modules',
        'venv',
        '.conda',
        'project-Mern',
        'mern-dashboard'
    ]
    
    for item in to_remove:
        try:
            path = Path(item)
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except Exception as e:
            print(f"Error removing {item}: {e}")

def create_gitkeep_files():
    """Create .gitkeep files in empty directories."""
    empty_dirs = [
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
        'scripts/maintenance'
    ]
    
    for directory in empty_dirs:
        try:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / '.gitkeep').touch()
        except Exception as e:
            print(f"Error creating .gitkeep in {directory}: {e}")

def main():
    """Main function to clean up the repository."""
    print("Starting repository cleanup...")
    
    # Move remaining files
    print("Moving remaining files...")
    move_files()
    
    # Create .gitkeep files
    print("Creating .gitkeep files...")
    create_gitkeep_files()
    
    # Clean up unnecessary directories
    print("Cleaning up unnecessary directories...")
    cleanup_directories()
    
    print("Repository cleanup complete!")

if __name__ == "__main__":
    main() 