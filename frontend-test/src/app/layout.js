export const metadata = {
  title: 'Test App',
  description: 'Minimal Next.js test',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
} 