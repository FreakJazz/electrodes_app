from pydantic import BaseModel

# data class
class CrossValidation(BaseModel):
    solver: str
    tol: str
    shrinkage: str
    
    class Config:
        orm_mode = True