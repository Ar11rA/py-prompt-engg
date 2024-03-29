from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": """
      Given the following SQL tables, your job is to write queries given a user’s request.\n    \n    
      CREATE TABLE Orders (\n      OrderID int,\n      CustomerID int,\n      OrderDate datetime,\n      OrderTime varchar(8),\n      PRIMARY KEY (OrderID)\n    );\n    \n    
      CREATE TABLE OrderDetails (\n      OrderDetailID int,\n      OrderID int,\n      ProductID int,\n      Quantity int,\n      PRIMARY KEY (OrderDetailID)\n    );\n    \n    
      CREATE TABLE Products (\n      ProductID int,\n      ProductName varchar(50),\n      Category varchar(50),\n      UnitPrice decimal(10, 2),\n      Stock int,\n      PRIMARY KEY (ProductID)\n    );\n    \n    
      CREATE TABLE Customers (\n      CustomerID int,\n      FirstName varchar(50),\n      LastName varchar(50),\n      Email varchar(100),\n      Phone varchar(20),\n      PRIMARY KEY (CustomerID)\n    );
      
      So if you have to query all columns for Orders table using SQL, the query is select * from Orders;
      
      Don't write any extra text in the answer other than the query
      """
        },
        {
            "role": "user",
            "content": """
      Write a SQL query which computes the average total order value for all orders on 2023-04-01.
      """
        }
    ],
    temperature=0.7,
    max_tokens=1000,
    top_p=1
)

print(response.choices[0].message.content)
