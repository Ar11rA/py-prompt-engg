from pydantic import BaseModel, Field


class Location(BaseModel):
    """Get location details such as latitude, longitude and short one liner description"""
    latitude: str = Field(description="The latitude of the location")
    longitude: str = Field(description="The longitude of the location")
    description: str = Field(description="The description of the given location")
