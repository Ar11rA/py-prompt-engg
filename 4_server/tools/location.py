from langchain_core.tools import tool
from pydantic import BaseModel, Field


class Location(BaseModel):
    """Get location details such as latitude, longitude and short one liner description"""
    latitude: str = Field(description="The latitude of the location")
    longitude: str = Field(description="The longitude of the location")
    description: str = Field(description="The description of the given location")


@tool(args_schema=Location, return_direct=True)
def get_location_coordinates(latitude: float, longitude: float, description: str) -> str:
    """Fetch coordinates."""
    return "The location co-ordinates: {0},{1} with desc: {2}"\
        .format(str(latitude), str(longitude), description)
