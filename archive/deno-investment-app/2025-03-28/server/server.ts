import { Application, Router, oakCors, DB, log, join, send } from "./deps.ts";

// Initialize logging
await log.setup({
  handlers: {
    console: new log.handlers.ConsoleHandler("INFO"),
  },
  loggers: {
    default: {
      level: "INFO",
      handlers: ["console"],
    },
  },
});

const logger = log.getLogger();

// Initialize database
const db = new DB("investment.db");
db.execute(`
  CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    recommendation TEXT NOT NULL,
    confidence REAL NOT NULL,
    target_price REAL NOT NULL,
    created_at TEXT NOT NULL
  )
`);

// Seed some data if table is empty
const count = db.query("SELECT COUNT(*) as count FROM recommendations")[0][0];
if (count === 0) {
  const stocks = [
    { symbol: "AAPL", name: "Apple Inc.", price: 187.92, recommendation: "Buy", confidence: 0.85, target_price: 210.50 },
    { symbol: "MSFT", name: "Microsoft Corp.", price: 420.01, recommendation: "Hold", confidence: 0.70, target_price: 425.00 },
    { symbol: "GOOGL", name: "Alphabet Inc.", price: 175.63, recommendation: "Buy", confidence: 0.80, target_price: 190.25 },
    { symbol: "AMZN", name: "Amazon.com Inc.", price: 180.25, recommendation: "Buy", confidence: 0.75, target_price: 200.00 },
    { symbol: "META", name: "Meta Platforms Inc.", price: 500.11, recommendation: "Hold", confidence: 0.60, target_price: 515.00 }
  ];

  for (const stock of stocks) {
    db.query(
      "INSERT INTO recommendations (symbol, name, price, recommendation, confidence, target_price, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
      [stock.symbol, stock.name, stock.price, stock.recommendation, stock.confidence, stock.target_price, new Date().toISOString()]
    );
  }
  
  logger.info("Database seeded with sample stock recommendations");
}

// Create the router
const router = new Router();

// Health check endpoint
router.get("/api/health", (ctx) => {
  ctx.response.body = {
    status: "healthy",
    api_version: "1.0.0",
    environment: process.env.ENVIRONMENT || "development"
  };
});

// Get all recommendations
router.get("/api/recommendations", (ctx) => {
  const recommendations = db.query("SELECT * FROM recommendations ORDER BY confidence DESC");
  const formattedRecommendations = recommendations.map(([id, symbol, name, price, recommendation, confidence, target_price, created_at]) => ({
    id,
    symbol,
    name,
    price,
    recommendation,
    confidence,
    target_price,
    created_at,
    potential_return: Number(((target_price - price) / price * 100).toFixed(2)),
  }));
  
  ctx.response.body = formattedRecommendations;
});

// Get recommendation by symbol
router.get("/api/recommendations/:symbol", (ctx) => {
  const symbol = ctx.params.symbol?.toUpperCase();
  if (!symbol) {
    ctx.response.status = 400;
    ctx.response.body = { error: "Symbol is required" };
    return;
  }
  
  const recommendation = db.query("SELECT * FROM recommendations WHERE symbol = ? LIMIT 1", [symbol]);
  
  if (recommendation.length === 0) {
    ctx.response.status = 404;
    ctx.response.body = { error: "Recommendation not found" };
    return;
  }
  
  const [id, sym, name, price, rec, confidence, target_price, created_at] = recommendation[0];
  
  ctx.response.body = {
    id,
    symbol: sym,
    name,
    price,
    recommendation: rec,
    confidence,
    target_price,
    created_at,
    potential_return: Number(((target_price - price) / price * 100).toFixed(2)),
  };
});

// Get portfolio summary
router.get("/api/portfolio/summary", (ctx) => {
  ctx.response.body = {
    total_value: 125400,
    monthly_growth: 5.2,
    yearly_growth: 12.8,
    risk_score: 68,
    sectors: {
      technology: 35,
      financial: 20,
      healthcare: 15,
      consumer: 10,
      energy: 10,
      other: 10
    },
    top_performers: [
      { symbol: "AAPL", name: "Apple Inc.", gain: 15.3 },
      { symbol: "MSFT", name: "Microsoft Corp.", gain: 12.1 },
      { symbol: "AMZN", name: "Amazon.com Inc.", gain: 9.8 }
    ],
    worst_performers: [
      { symbol: "INTC", name: "Intel Corp.", loss: -5.2 },
      { symbol: "IBM", name: "IBM Corp.", loss: -3.1 },
      { symbol: "CSCO", name: "Cisco Systems Inc.", loss: -1.8 }
    ]
  };
});

// Model performance analytics
router.get("/api/models/performance", (ctx) => {
  ctx.response.body = {
    models: [
      { name: "LSTM", accuracy: 86.7, trend: "+2.1%" },
      { name: "ARIMA", accuracy: 82.3, trend: "+1.5%" },
      { name: "Prophet", accuracy: 79.8, trend: "-0.3%" }
    ],
    last_updated: new Date().toISOString()
  };
});

// Create the application
const app = new Application();

// Add middleware
app.use(oakCors());

// Add logging middleware
app.use(async (ctx, next) => {
  const start = Date.now();
  await next();
  const ms = Date.now() - start;
  logger.info(`${ctx.request.method} ${ctx.request.url.pathname} ${ctx.response.status} ${ms}ms`);
});

// Add router for API endpoints
app.use(router.routes());
app.use(router.allowedMethods());

// Serve static files from the client/dist directory in development
const clientDistPath = join(process.cwd(), "../client/dist");

// For production, we would use the same approach but with a production build
app.use(async (ctx, next) => {
  try {
    const path = ctx.request.url.pathname;
    
    // If the request is for an API endpoint, move to the next middleware
    if (path.startsWith("/api")) {
      await next();
      return;
    }
    
    // Try to serve static files
    await send(ctx, path, {
      root: clientDistPath,
      index: "index.html",
    });
  } catch {
    // If the file doesn't exist, serve index.html for SPA routing
    try {
      await send(ctx, "/", {
        root: clientDistPath,
        index: "index.html",
      });
    } catch (error) {
      logger.error(`Error serving static files: ${error.message}`);
      await next();
    }
  }
});

// Start the server
const port = 8000;
logger.info(`Server running on http://localhost:${port}`);

await app.listen({ port }); 