"""
The below example will use a SQLite connection with the Chinook database, which is a sample database that represents a digital media store. Follow these installation steps to create Chinook.db in the same directory as this notebook. You can also download and build the database via the command line:

```bash
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db

```

Afterwards, place `Chinook.db` in the same directory where this code is running.

"""

from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import chain
# replace this with the connection details of your db
from langchain_openai import ChatOpenAI
import re

db = SQLDatabase.from_uri("sqlite:///ebook.db")
print(db.get_usable_table_names())
llm = ChatOpenAI(temperature=0, model="qwen3:4b", base_url="http://localhost:11434/v1", api_key="123")

# convert question to sql query
write_query = create_sql_query_chain(llm, db)

@chain
def filter_sql(sql):
    #remove contents between <think>...</think>
    sql = re.sub(r"<think>.*?</think>", "", sql, flags=re.DOTALL).strip()
    sql = sql.replace("SQLQuery:", "").strip()
    print(f"Using SQL: {sql}")
    return sql

# Execute SQL query
execute_query = QuerySQLDatabaseTool(db=db)

# combined chain = write_query | execute_query
combined_chain = write_query | filter_sql | execute_query

# run the chain
result = combined_chain.invoke({"question": "How many ebooks are there?"})

print(result)
