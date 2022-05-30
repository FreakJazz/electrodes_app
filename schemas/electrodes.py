from pydantic import BaseModel

# data class
class Electrodes(BaseModel):
    Date: str
    E1:  float
    E2:  float
    E3:  float
    E4:  float
    E5:  float
    E6:  float
    E7:  float
    E8:  float
    E9:  float
    E10: float
    E11: float
    E12: float
    E13: float
    E14: float
    E15: float
    E16: float
    E17: float
    E18: float
    E19: float
    E20: float
    E21: float
    E22: float
    E23: float
    E24: float
    E25: float
    E26: float
    E27: float
    E28: float
    E29: float
    E30: float
    E31: float
    E32: float

    class Config:
        orm_mode = True