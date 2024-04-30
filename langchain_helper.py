from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector, FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain.prompts.prompt import PromptTemplate
from IPython.display import display, Markdown
import pathlib
import textwrap
# You might not necessarily need to import few_shots (remove if not used)
# from few_shots import few_shots  # Import few_shots data


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def get_few_shot_db_chain():
    # Define database credentials directly
    db_user = "root"
    db_password = "123456789"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)

    # Pre-built vectorstore path (replace with your actual path)
    vectorstore_path = "path/to/your/vectorstore.lvm"

    # Load pre-built vectorstore (optional, comment out if not used)
    # vectorstore = Chroma.load(vectorstore_path)

    example_selector = SemanticSimilarityExampleSelector(
        # Replace with your vectorstore if using (uncomment the above line)
        # vectorstore=vectorstore,
        k=2,  # Adjust k as necessary
    )

    mysql_prompt = """
    You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. Use the following format:

    Question: 
    SQLQuery: 
    SQLResult: 
    Answer: 
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],  # Variables used in the prompt
    )

    # Define tokenizer and model for sentence encoding (replace with your choices)
    model_name = "facebook/bart-base"  # Example model for text generation
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    def generate_sql_from_text(text):
        # Use transformers-bart-sql for text-to-SQL conversion
        from transformers import BartForConditionalGeneration, BartTokenizer
        bart_sql_model_name = "bart-sql/bart-base-sql"
        bart_tokenizer = BartTokenizer.from_pretrained(bart_sql_model_name)
        bart_model = BartForConditionalGeneration.from_pretrained(bart_sql_model_name)

        input_ids = bart_tokenizer(text, return_tensors="pt")["input_ids"]
        generated_ids = bart_model.generate(input_ids)
        output = bart_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return output.strip()

    chain = SQLDatabaseChain.from_generative_model(None, db, verbose=True, prompt=few_shot_prompt)
    return chain


# Example usage (assuming you have processed data in 'few_shots')
chain = get_few_shot_db_chain()
user_input = "Find all products with a price greater than $100"

# Generate descriptive text using the model (assuming you can adapt Gemini for this task
