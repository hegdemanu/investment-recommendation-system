from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps

router = APIRouter()


@router.get("/", response_model=List[schemas.Portfolio])
def read_portfolios(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve portfolios.
    """
    portfolios = crud.portfolio.get_multi_by_owner(
        db=db, owner_id=current_user.id, skip=skip, limit=limit
    )
    return portfolios


@router.post("/", response_model=schemas.Portfolio)
def create_portfolio(
    *,
    db: Session = Depends(deps.get_db),
    portfolio_in: schemas.PortfolioCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Create new portfolio.
    """
    portfolio = crud.portfolio.create_with_owner(
        db=db, obj_in=portfolio_in, owner_id=current_user.id
    )
    return portfolio


@router.get("/{id}", response_model=schemas.Portfolio)
def read_portfolio(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Get portfolio by ID.
    """
    portfolio = crud.portfolio.get(db=db, id=id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if portfolio.owner_id != current_user.id:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    return portfolio


@router.put("/{id}", response_model=schemas.Portfolio)
def update_portfolio(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    portfolio_in: schemas.PortfolioUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Update a portfolio.
    """
    portfolio = crud.portfolio.get(db=db, id=id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if portfolio.owner_id != current_user.id:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    portfolio = crud.portfolio.update(db=db, db_obj=portfolio, obj_in=portfolio_in)
    return portfolio


@router.delete("/{id}", response_model=schemas.Portfolio)
def delete_portfolio(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Delete a portfolio.
    """
    portfolio = crud.portfolio.get(db=db, id=id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if portfolio.owner_id != current_user.id:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    portfolio = crud.portfolio.remove(db=db, id=id)
    return portfolio 