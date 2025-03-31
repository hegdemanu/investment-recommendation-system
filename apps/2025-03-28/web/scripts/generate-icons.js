const fs = require('fs');
const { createCanvas } = require('canvas');

// Function to create a simple icon with text
function createIcon(size, text, filename) {
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');
  
  // Fill with a gradient background
  const gradient = ctx.createLinearGradient(0, 0, size, size);
  gradient.addColorStop(0, '#3182ce');
  gradient.addColorStop(1, '#2468ab');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  
  // Add text
  ctx.fillStyle = 'white';
  ctx.font = `bold ${size/4}px Arial`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, size/2, size/2);
  
  // Save the image
  const buffer = canvas.toBuffer('image/png');
  fs.writeFileSync(filename, buffer);
  console.log(`Created ${filename}`);
}

// Create icons directory if it doesn't exist
const iconsDir = './public/icons';
if (!fs.existsSync(iconsDir)) {
  fs.mkdirSync(iconsDir, { recursive: true });
}

// Create various icon sizes
createIcon(192, 'IRS', `${iconsDir}/icon-192x192.png`);
createIcon(512, 'IRS', `${iconsDir}/icon-512x512.png`);
createIcon(384, 'IRS', `${iconsDir}/icon-384x384.png`);
createIcon(256, 'IRS', `${iconsDir}/icon-256x256.png`);
createIcon(128, 'IRS', `${iconsDir}/icon-128x128.png`);
createIcon(96, 'IRS', `${iconsDir}/icon-96x96.png`);
createIcon(72, 'IRS', `${iconsDir}/icon-72x72.png`);
createIcon(180, 'IRS', `${iconsDir}/apple-icon-180.png`);

console.log('All icons generated successfully!'); 