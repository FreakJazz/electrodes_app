from pydantic import BaseModel

# data class
class Electrodes(BaseModel):
    Date: str
    E1:  str
    E2:  str
    E3:  str
    E4:  str
    E5:  str
    E6:  str
    E7:  str
    E8:  str
    E9:  str
    E10: str
    E11: str
    E12: str
    E13: str
    E14: str
    E15: str
    E16: str
    E17: str
    E18: str
    E19: str
    E20: str
    E21: str
    E22: str
    E23: str
    E24: str
    E25: str
    E26: str
    E27: str
    E28: str
    E29: str
    E30: str
    E31: str
    E32: str

    class Config:
        orm_mode = True