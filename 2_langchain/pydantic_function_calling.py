from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# without explicit formatting
model = ChatOpenAI()
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    """ 
    Tell me the hex code for the input color: {color} 
    Also, let me know of a good description for the color. 
    Include usage in a sentence in the description.
    """,
)

chain = prompt | model | output_parser

print(chain.invoke({"color": "cyan"}))


# with explicit formatting
class ColorDetails(BaseModel):
    """Call this to export color details"""
    hexCode: str = Field(description="Hex code corresponding to the color")
    colorText: str = Field(description="The color text")
    description: str = Field(description="The color description and usage")


output_parser = PydanticOutputParser(pydantic_object=ColorDetails)

prompt = ChatPromptTemplate.from_template(
    """ 
    Tell me the hex code for the input color: {color} 
    {instructions}
    """,
)

# color_function = convert_pydantic_to_openai_function(ColorDetails)
model_with_fn = ChatOpenAI()
print(output_parser.get_format_instructions())
chain = prompt | model_with_fn | output_parser
res = chain.invoke({"color": "cyan", "instructions": output_parser.get_format_instructions()})
print(type(res))
