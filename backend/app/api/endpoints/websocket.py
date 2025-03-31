from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import List, Dict, Any
import asyncio
import json
import logging
from datetime import datetime
import pandas as pd

from app.db.session import get_db
from app.core.config import settings
from sqlalchemy.orm import Session
from app.models.ticker import Ticker
from app.models.price_data import PriceData
from app.services.data_service import DataService

router = APIRouter()

# ConnectionManager to handle multiple client connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.logger = logging.getLogger(__name__)
        
    async def connect(self, websocket: WebSocket, client_id: str, ticker: str = None):
        await websocket.accept()
        
        # Store connection based on ticker subscription
        if ticker:
            if ticker not in self.active_connections:
                self.active_connections[ticker] = []
            self.active_connections[ticker].append(websocket)
            self.logger.info(f"Client {client_id} connected to {ticker} updates")
        else:
            # General connection not specific to a ticker
            if "general" not in self.active_connections:
                self.active_connections["general"] = []
            self.active_connections["general"].append(websocket)
            self.logger.info(f"Client {client_id} connected to general updates")
    
    def disconnect(self, websocket: WebSocket, ticker: str = None):
        # Remove connection from appropriate list
        if ticker and ticker in self.active_connections:
            if websocket in self.active_connections[ticker]:
                self.active_connections[ticker].remove(websocket)
                self.logger.info(f"Client disconnected from {ticker} updates")
        else:
            # Check general connections
            if "general" in self.active_connections and websocket in self.active_connections["general"]:
                self.active_connections["general"].remove(websocket)
                self.logger.info("Client disconnected from general updates")
    
    async def broadcast_to_ticker(self, ticker: str, message: Dict[str, Any]):
        if ticker in self.active_connections:
            disconnected_websockets = []
            for websocket in self.active_connections[ticker]:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    self.logger.error(f"Error sending to websocket: {e}")
                    disconnected_websockets.append(websocket)
            
            # Remove any disconnected websockets
            for websocket in disconnected_websockets:
                self.disconnect(websocket, ticker)
    
    async def broadcast_general(self, message: Dict[str, Any]):
        if "general" in self.active_connections:
            disconnected_websockets = []
            for websocket in self.active_connections["general"]:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    self.logger.error(f"Error sending to websocket: {e}")
                    disconnected_websockets.append(websocket)
            
            # Remove any disconnected websockets
            for websocket in disconnected_websockets:
                self.disconnect(websocket)

# Create connection manager instance
manager = ConnectionManager()
data_service = DataService()

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    client_id: str, 
    db: Session = Depends(get_db)
):
    """General WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()
            
            # Parse the message
            try:
                message = json.loads(data)
                action = message.get("action", "")
                
                # Handle different actions
                if action == "ping":
                    await websocket.send_json({"action": "pong", "timestamp": datetime.now().isoformat()})
                elif action == "subscribe":
                    # Handle subscription request
                    ticker = message.get("ticker")
                    if ticker:
                        # Move connection to ticker-specific group
                        manager.disconnect(websocket)
                        await manager.connect(websocket, client_id, ticker)
                        await websocket.send_json({"action": "subscribed", "ticker": ticker})
                        
                        # Send initial data for the ticker
                        initial_data = data_service.get_latest_price_data(db, ticker)
                        if initial_data:
                            await websocket.send_json({
                                "action": "data",
                                "ticker": ticker,
                                "data": initial_data
                            })
                else:
                    await websocket.send_json({"action": "error", "message": "Unknown action"})
                    
            except json.JSONDecodeError:
                await websocket.send_json({"action": "error", "message": "Invalid JSON"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@router.websocket("/ws/ticker/{ticker}/{client_id}")
async def ticker_websocket_endpoint(
    websocket: WebSocket, 
    ticker: str, 
    client_id: str, 
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for ticker-specific real-time updates"""
    # Check if ticker exists in database
    db_ticker = db.query(Ticker).filter(Ticker.symbol == ticker).first()
    if not db_ticker:
        await websocket.accept()
        await websocket.send_json({"action": "error", "message": f"Ticker {ticker} not found"})
        await websocket.close()
        return
    
    await manager.connect(websocket, client_id, ticker)
    
    try:
        # Send initial data
        initial_data = data_service.get_latest_price_data(db, ticker)
        if initial_data:
            await websocket.send_json({
                "action": "data",
                "ticker": ticker,
                "data": initial_data
            })
        
        # Listen for client messages
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                action = message.get("action", "")
                
                if action == "ping":
                    await websocket.send_json({"action": "pong", "timestamp": datetime.now().isoformat()})
                elif action == "unsubscribe":
                    manager.disconnect(websocket, ticker)
                    await manager.connect(websocket, client_id)  # Move back to general
                    await websocket.send_json({"action": "unsubscribed", "ticker": ticker})
                    break
                    
            except json.JSONDecodeError:
                await websocket.send_json({"action": "error", "message": "Invalid JSON"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, ticker)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, ticker)

# Background task to regularly send updates to connected clients
async def price_update_task():
    """Background task to send periodic price updates to websocket clients"""
    db = next(get_db())
    
    while True:
        try:
            # Get all tickers being tracked with active connections
            active_tickers = list(manager.active_connections.keys())
            active_tickers = [t for t in active_tickers if t != "general"]
            
            for ticker in active_tickers:
                # Get latest data for the ticker
                latest_data = data_service.get_latest_price_data(db, ticker)
                
                if latest_data:
                    # Broadcast to all clients subscribed to this ticker
                    await manager.broadcast_to_ticker(ticker, {
                        "action": "update",
                        "ticker": ticker,
                        "data": latest_data,
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Send a summary to general subscribers
            if "general" in manager.active_connections and manager.active_connections["general"]:
                summary = data_service.get_market_summary(db)
                await manager.broadcast_general({
                    "action": "market_summary",
                    "data": summary,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            logging.error(f"Error in price update task: {e}")
            
        # Wait before sending next update
        await asyncio.sleep(settings.WEBSOCKET_UPDATE_INTERVAL)  # Update interval in seconds

# Start the background task when the application starts
@router.on_event("startup")
async def startup_event():
    asyncio.create_task(price_update_task()) 