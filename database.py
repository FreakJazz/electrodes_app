import os


AIRTABLE_BASE_ID=os.environ.get("BASE_ID")
AIRTABLE_API_KEY=os.environ.get("API_KEY")
AIRTABLE_TABLE_NAME=os.environ.get("TABLE_NAME")

# SQLALCHEMY_DATABASE_URL = f'https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}'
# SQLALCHEMY_DATABASE_URL = f"airtable://:keyXXXX@appYYY?peek_rows=10&tables=tableA&tables=tableB"

endpoint = f'https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}'

# engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True, future=True)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()