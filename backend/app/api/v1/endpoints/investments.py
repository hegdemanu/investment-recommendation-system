from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps

router = APIRouter()


@router.get("/{portfolio_id}/", response_model=List[schemas.Investment])
def read_investments(
    portfolio_id: int,
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve investments in a portfolio.
    """
    portfolio = crud.portfolio.get(db=db, id=portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if portfolio.owner_id != current_user.id:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    
    investments = crud.investment.get_multi_by_portfolio(
        db=db, portfolio_id=portfolio_id, skip=skip, limit=limit
    )
    return investments


@router.post("/{portfolio_id}/", response_model=schemas.Investment)
def create_investment(
    *,
    db: Session = Depends(deps.get_db),
    portfolio_id: int,
    investment_in: schemas.InvestmentCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Create new investment in a portfolio.
    """
    portfolio = crud.portfolio.get(db=db, id=portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if portfolio.owner_id != current_user.id:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    
    # Check if investment with this symbol already exists
    existing_investment = crud.investment.get_by_symbol(
        db=db, portfolio_id=portfolio_id, symbol=investment_in.symbol
    )
    if existing_investment:
        raise HTTPException(
            status_code=400,
            detail=f"Investment with symbol {investment_in.symbol} already exists in this portfolio",
        )
    
    investment = crud.investment.create_with_portfolio(
        db=db, obj_in=investment_in, portfolio_id=portfolio_id
    )
    return investment


@router.put("/{portfolio_id}/{id}", response_model=schemas.Investment)
def update_investment(
    *,
    db: Session = Depends(deps.get_db),
    portfolio_id: int,
    id: int,
    investment_in: schemas.InvestmentUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Update an investment in a portfolio.
    """
    portfolio = crud.portfolio.get(db=db, id=portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if portfolio.owner_id != current_user.id:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    
    investment = crud.investment.get(db=db, id=id)
    if not investment:
        raise HTTPException(status_code=404, detail="Investment not found")
    if investment.portfolio_id != portfolio_id:
        raise HTTPException(status_code=400, detail="Investment not in this portfolio")
    
    investment = crud.investment.update(db=db, db_obj=investment, obj_in=investment_in)
    return investment


@router.delete("/{portfolio_id}/{id}", response_model=schemas.Investment)
def delete_investment(
    *,
    db: Session = Depends(deps.get_db),
    portfolio_id: int,
    id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Delete an investment from a portfolio.
    """
    portfolio = crud.portfolio.get(db=db, id=portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if portfolio.owner_id != current_user.id:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    
    investment = crud.investment.get(db=db, id=id)
    if not investment:
        raise HTTPException(status_code=404, detail="Investment not found")
    if investment.portfolio_id != portfolio_id:
        raise HTTPException(status_code=400, detail="Investment not in this portfolio")
    
    investment = crud.investment.remove(db=db, id=id)
    return investment 