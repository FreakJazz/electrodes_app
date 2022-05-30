from pydantic import BaseModel

# data class
class Porcentage(BaseModel):
    solver: str
    tol: str
    shrinkage: str
    test_size: str
    
    class Config:
        orm_mode = True