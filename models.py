from sqlalchemy import Column, Integer, String, Numeric, Boolean, Timestamp, func
from .database import Base  # Links directly to new database file!


class UnfallModel(Base):
    __tablename__ = "unfaelle"

    id = Column(Integer, primary_key=True, index=True)
    roh_beschreibung = Column(String, nullable=False)
    auto_marke = Column(String(50))
    auto_baujahr = Column(Integer)
    beschaedigte_teile = Column(String)
    interner_schadensgrad = Column(String(20))
    vorhergesagte_reparaturkosten = Column(Numeric(10, 2))
    ist_totalschaden = Column(Boolean, default=False)
    erstellt_am = Column(Timestamp, server_default=func.now())

    # The 3 tracking columns built for the AI automation pipeline!
    unfall_bild_url = Column(String(512), nullable=True)
    audio_text_extrahiert = Column(Boolean, default=False)
    verarbeitungs_status = Column(String(20), default="ausstehend")
