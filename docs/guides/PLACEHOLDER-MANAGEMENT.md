# Placeholder Management System

This system helps you track and manage all placeholders in the project that need to be filled before the application can function properly.

## Overview

The Investment Recommendation System requires various configuration values, API keys, and database setups to function correctly. This placeholder management system tracks all these requirements in a centralized way and can automatically apply them to your configuration files.

## Files

1. **placeholders.json** - The central registry of all placeholders in the project
2. **update-placeholders.js** - A utility script for managing placeholder statuses and values
3. **apply-placeholders.js** - A utility script that applies values from placeholders.json to actual configuration files
4. **PLACEHOLDER-MANAGEMENT.md** - This documentation file

## Using the Placeholder Management Tool

1. Make the scripts executable:
   ```bash
   chmod +x update-placeholders.js apply-placeholders.js
   ```

2. Run the utility script:
   ```bash
   node update-placeholders.js
   ```

3. The interactive menu will allow you to:
   - List all placeholders and their status
   - Mark placeholders as filled or unfilled
   - Set actual values for placeholders
   - Apply values from placeholders.json to actual configuration files

## Placeholder Structure

Each placeholder in `placeholders.json` follows this structure:

```json
{
  "path.to.placeholder": {
    "placeholder": "example-value",
    "value": "actual-value-to-use",
    "description": "Description of what this value is for",
    "required": true,
    "location": "path/to/file",
    "status": "unfilled"
  }
}
```

- **placeholder**: Example or template value
- **value**: The actual value to use (this is what gets applied to files)
- **description**: What this value is used for
- **required**: Whether this is required for the application to function
- **location**: Where this value needs to be set
- **status**: Current status ("filled" or "unfilled")

## Managing Placeholders

### Setting Values

1. Run `node update-placeholders.js`
2. Select option 4: "Set value for placeholder"
3. Enter the path to the placeholder (e.g., `auth.supabase.NEXT_PUBLIC_SUPABASE_URL`)
4. Enter the actual value to use

### Applying Values to Configuration Files

After setting values in placeholders.json, you can apply them to your actual configuration files:

1. Run `node update-placeholders.js`
2. Select option 5: "Apply placeholder values to configuration files"
3. Confirm the operation when prompted

Alternatively, you can run the apply script directly:

```bash
node apply-placeholders.js
```

You can also run in dry-run mode to see what would be changed without making actual modifications:

```bash
node apply-placeholders.js --dry-run
```

## Requirements Overview

The system tracks the following categories of placeholders:

1. **Authentication (Supabase)**
   - Supabase URL and anon key
   - Test user credentials
   - Database schema

2. **API Configuration**
   - API URL
   - Server port

3. **Deployment Settings**
   - Render API key and service ID
   - Vercel project name and team ID

4. **Backend Configuration**
   - Database connection strings
   - JWT secrets
   - Server ports

5. **FastAPI Configuration**
   - Database URL
   - JWT secret

6. **Dependencies**
   - Required packages

## How to Update When Adding New Placeholders

When you add new configuration requirements to the project:

1. Edit `placeholders.json` to add the new placeholder with its details
2. Update this documentation if needed
3. Run the utility script to verify your changes

## Verification Process

Before considering the setup complete, ensure all required placeholders are marked as "filled" using the placeholder management tool. 