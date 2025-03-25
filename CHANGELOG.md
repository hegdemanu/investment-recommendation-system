# Changelog

## [1.0.0] - 2023

### Added
- Restructured the project into a more modular and maintainable architecture
- Created separate modules for core logic, dashboard, API, and utilities
- Implemented a centralized configuration system
- Added sample data generation for demonstration purposes
- Improved the dashboard with a responsive design and dark mode toggle
- Added JSON viewer for detailed examination of report data
- Implemented better error handling for file operations

### Changed
- Moved dashboard generation to a dedicated module
- Unified the command-line interface with a single entry point script
- Enhanced CSS styling with responsive design and improved theming
- Improved the file loading process with better error handling

### Fixed
- Fixed formatting issues with braces in the HTML template
- Fixed JSON viewing for local file URLs by using XMLHttpRequest instead of fetch API
- Improved the report generation process with better error handling
- Enhanced the filter functionality for data files by type and format 