import openai
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from vars import API_KEY

openai.api_key = API_KEY

# without explicit formatting
model = ChatOpenAI()
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    """ 
    Tell me the hex code for the input color: {color} 
    Also, let me know of a good description for the color. 
    Include usage in a sentence in the description.
    """
)

chain = prompt | model | output_parser

print(chain.invoke({"color": "cyan"}))


# with explicit formatting
class ColorDetails(BaseModel):
    """Call this to export color details"""
    hexCode: str = Field(description="Hex code corresponding to the color")
    colorText: str = Field(description="The color text")
    description: str = Field(description="The color description and usage")


color_function = convert_pydantic_to_openai_function(ColorDetails)
model_with_fn = ChatOpenAI().bind(functions=[color_function])
chain = prompt | model_with_fn
print(chain.invoke({"color": "cyan"}))
