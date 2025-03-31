# Placeholders to Fill

This document lists all placeholders that need to be filled in the repository before the application can function properly. For a more interactive way to manage placeholders, use the placeholder management system.

> **Note:** We now have a better way to manage placeholders! Use the placeholder management tool by running `node update-placeholders.js` from the project root.

## 1. Supabase Configuration

### In `client/.env.local`:
- `NEXT_PUBLIC_SUPABASE_URL`: Replace with your actual Supabase project URL
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`: Replace with your actual Supabase anonymous key

### Database Setup:
- Run the SQL schema in `client/README-AUTH.md` to set up required tables
- Create admin and test users as outlined in the README-AUTH.md document

## 2. API Configuration

### In `client/.env.local`:
- `NEXT_PUBLIC_API_URL`: URL for your backend API (default is http://localhost:5001/api)

## 3. Deployment Configuration (Optional)

### For Render Deployment:
- `RENDER_API_KEY`: Your Render API key for CI/CD
- `RENDER_SERVICE_ID`: Your Render service ID

### For Vercel Deployment:
- `VERCEL_PROJECT_NAME`: Your Vercel project name
- `VERCEL_TEAM_ID`: Your Vercel team ID

## 4. Test User Credentials

Create these users in your Supabase project:

- Admin User:
  - Email: admin@example.com
  - Password: Admin123!
  - Custom user metadata: `{"role": "admin"}`

- Regular User:
  - Email: user@example.com
  - Password: User123!

## 5. Backend Configuration

The server at `/server/server.js` may need additional configuration:

- Database connection strings
- API keys for any external services
- Environment-specific settings

## How to Confirm Setup is Complete

1. Fill in all the placeholders listed above
2. Run both frontend and backend applications
3. Test authentication flows:
   - Register a new user
   - Login as admin
   - Login as regular user
   - Test protected routes
4. If all features work correctly, your setup is complete

## Using the Placeholder Management System

For better tracking of your progress filling placeholders, use our management system:

1. See all placeholders in the JSON format in `placeholders.json`
2. Run the utility script to interactively manage placeholders:
   ```bash
   node update-placeholders.js
   ```
3. Mark placeholders as "filled" as you complete them

See [PLACEHOLDER-MANAGEMENT.md](PLACEHOLDER-MANAGEMENT.md) for detailed documentation. 