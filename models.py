from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class PassengerData(db.Model):
    __tablename__ = "passenger_data"

    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    passengers = db.Column(db.Integer, nullable=False)

    # уникальность пары (год, месяц), чтобы можно было ссылаться
    __table_args__ = (
        db.UniqueConstraint("year", "month", name="passenger_data_year_month_uk"),
    )

    # связи
    predictions = db.relationship(
        "Prediction",
        back_populates="period",
        lazy="dynamic",
    )
    incidents = db.relationship(
        "Incident",
        back_populates="period",
        lazy="dynamic",
    )


class User(db.Model):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)

    # связь с прогнозами
    predictions = db.relationship(
        "Prediction",
        back_populates="user",
        lazy="dynamic",
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    predicted_passengers = db.Column(db.Float, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    # 🔗 кто запустил прогноз
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=True,
    )

    # 🔗 на какой период (год+месяц) строится прогноз
    __table_args__ = (
        db.ForeignKeyConstraint(
            ["year", "month"],
            ["passenger_data.year", "passenger_data.month"],
            name="prediction_passenger_data_fk",
            onupdate="CASCADE",
            ondelete="RESTRICT",
        ),
    )

    # ORM-связи
    user = db.relationship("User", back_populates="predictions")
    period = db.relationship("PassengerData", back_populates="predictions")


class Incident(db.Model):
    __tablename__ = "incident"

    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    duration = db.Column(db.Integer, nullable=False)
    impact = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(255))

    __table_args__ = (
        db.ForeignKeyConstraint(
            ["year", "month"],
            ["passenger_data.year", "passenger_data.month"],
            name="incident_passenger_data_fk",
            onupdate="CASCADE",
            ondelete="RESTRICT",
        ),
    )

    # к какому периоду относится инцидент
    period = db.relationship("PassengerData", back_populates="incidents")
