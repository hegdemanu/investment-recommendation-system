# Archived Launcher Scripts

This directory contains launcher scripts that were previously used for the Investment Recommendation System. These have been archived in favor of the simpler, platform-independent `launch.py` script in the root directory.

## Files in this directory:

* `launcher.sh` - Simple bash script for Linux/macOS
* `launcher.bat` - Simple batch script for Windows
* `launcher.command` - Enhanced macOS launcher (clickable from Finder)
* `RunInvestmentSystem.app.command` - Fancy macOS launcher with colored output
* `run.sh` - Alternative bash script for Linux/macOS
* `run.bat` - Alternative batch script for Windows
* `conda_run.sh` - Specialized script for conda environments

## Why these are archived

These scripts have been replaced by the more maintainable `launch.py` script in the project root, which:
- Works on all platforms (Windows, macOS, Linux)
- Handles different Python installations automatically
- Shows better error messages
- Requires fewer files to maintain

## If you need these files

While not recommended, if you need to use one of these legacy launchers:
1. Copy the file back to the project root directory
2. Make it executable (for shell scripts on Linux/macOS):
   ```bash
   chmod +x script_name.sh
   ```
3. Run it according to your platform 