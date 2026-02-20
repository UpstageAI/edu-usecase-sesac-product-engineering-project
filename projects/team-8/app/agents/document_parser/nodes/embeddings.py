from dotenv import load_dotenv

from langchain_upstage import UpstageEmbeddings

from rich import print as rprint

load_dotenv()


def load_underlying_embeddings():
    underlying_embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    # rprint(">>> underlying_embeddings", underlying_embeddings)
    return underlying_embeddings
