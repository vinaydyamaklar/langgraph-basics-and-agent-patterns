from pydantic import BaseModel, Field, conint
from datetime import date


class HotelBooking(BaseModel):
    """
    Represents a hotel room booking.
    """
    full_name: str = Field(..., description="Hotel guest name")
    checkin_data: date = Field(..., description="Checkin date for the hotel room")
    checkout_data: date = Field(..., description="Checkout date of the hotel room")
    number_of_rooms: conint(ge=1)
    number_of_guests: conint(ge=1)
    special_request: str = Field(..., description="Special user request")