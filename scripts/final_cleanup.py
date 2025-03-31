#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def move_remaining_files():
    """Move remaining files to their correct locations."""
    moves = [
        # Move Python files to appropriate locations
        ('run.py', 'backend/main.py'),
        ('run_investment_system.py', 'trading_engine/main.py'),
        ('train_sentiment_models.py', 'trading_engine/models/sentiment.py'),
        ('launch.py', 'scripts/deployment/launch.py'),
        ('setup_modular_dashboard.py', 'scripts/setup/setup_dashboard.py'),
        
        # Move configuration files
        ('config/*', 'backend/config/'),
        ('anaconda-project.yml', 'backend/'),
        ('pytest.ini', 'backend/'),
        ('setup.py', 'backend/'),
        
        # Move documentation files
        ('CHANGELOG.md', 'docs/'),
        ('REPOSITORY-STRUCTURE.md', 'docs/architecture/'),
        ('PLACEHOLDER-MANAGEMENT.md', 'docs/guides/'),
        ('PLACEHOLDERS.md', 'docs/guides/'),
        ('README.md', '.'),
        
        # Move log files
        ('*.log', 'data_pipeline/logs/'),
        
        # Move test files
        ('tests/*', 'backend/tests/'),
        
        # Move environment files
        ('.env', '.'),
        ('.env.template', '.'),
        
        # Move package files
        ('package.json', '.'),
        ('package-lock.json', '.'),
        ('pnpm-lock.yaml', '.'),
        ('requirements.txt', '.'),
        
        # Move Docker files
        ('docker/*', 'docker/'),
        ('docker-compose.yml', '.')
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

def cleanup_remaining():
    """Remove unnecessary files and directories."""
    to_remove = [
        'config',
        'packages',
        '.DS_Store',
        'MANIFEST.in',
        'turbo.json',
        'pnpm-workspace.yaml',
        'setup-complete-system.js',
        'apply-placeholders.js',
        'update-placeholders.js',
        'placeholders.json'
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

def main():
    """Main function for final cleanup."""
    print("Starting final cleanup...")
    
    # Move remaining files
    print("Moving remaining files...")
    move_remaining_files()
    
    # Clean up unnecessary files
    print("Cleaning up unnecessary files...")
    cleanup_remaining()
    
    print("Final cleanup complete!")

if __name__ == "__main__":
    main() 