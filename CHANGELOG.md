# Changelog

## [1.1.0] - 2025

### Added
- Added detailed diversification information modal to the dashboard
- Improved information display in the pie chart section with subtitle and info icon
- Added clickable "more" indicator for expanding legend items in mobile view
- Implemented cursor sticking fixes for better user interaction
- Added documentation explaining dashboard features
- Added dedicated JSON directory in dashboard for better organization

### Changed
- Streamlined the README with more concise instructions
- Updated repository structure documentation
- Archived redundant launcher scripts
- Improved dashboard CSS and JavaScript for mobile responsiveness
- Enhanced dashboard generator to copy JSON files to dedicated directory
- Updated model documentation to clearly describe all three models (LSTM, GRU, Ensemble)

### Fixed
- Fixed pie chart rendering in mobile view
- Fixed cursor sticking issues with interactive elements
- Fixed template variable conflicts in dashboard generation
- Fixed legend display in mobile/popup mode

## [1.0.0] - 2025

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