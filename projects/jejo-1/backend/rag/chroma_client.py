import chromadb
import os
import shutil
from chromadb.utils import embedding_functions
from backend.utils.logger import get_logger

# Configuration
VECTOR_DB_DIR = "backend/vectorstore"
COLLECTION_NAME = "trade_coaching_knowledge"

# Ensure the vectorstore directory exists
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Initialize Chroma client
_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
logger = get_logger(__name__)


def get_or_create_collection():
    """
    Retrieves an existing Chroma collection or creates a new one if it doesn't exist.
    """
    logger.debug(
        "Attempting to get or create collection: %s in %s",
        COLLECTION_NAME,
        VECTOR_DB_DIR,
    )
    collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    logger.debug("Successfully got or created collection: %s", COLLECTION_NAME)
    return collection


def reset_collection():
    """
    Deletes the existing Chroma collection and recreates it.
    This effectively clears the vector database for a fresh ingestion.
    """
    logger.info("Resetting collection: %s in %s", COLLECTION_NAME, VECTOR_DB_DIR)
    try:
        # Attempt to delete the collection if it exists
        _client.delete_collection(name=COLLECTION_NAME)
        logger.info("Collection '%s' deleted successfully.", COLLECTION_NAME)
    except Exception as e:
        logger.warning("Collection '%s' delete skipped: %s", COLLECTION_NAME, e)
    
    # Recreate the collection
    collection = _client.create_collection(name=COLLECTION_NAME)
    logger.info("Collection '%s' recreated successfully.", COLLECTION_NAME)
    return collection

if __name__ == '__main__':
    print("--- Chroma Client Test ---")
    
    # Test reset and creation
    print("\n--- Testing reset_collection ---")
    collection_reset = reset_collection()
    print(f"Reset collection count: {collection_reset.count()}")

    # Test get_or_create_collection
    print("\n--- Testing get_or_create_collection ---")
    collection_get = get_or_create_collection()
    print(f"Get or created collection count: {collection_get.count()}")

    # Add some dummy data and count
    print("\n--- Adding dummy data ---")
    collection_get.add(
        documents=["This is a test document", "This is another test document"],
        metadatas=[{"source": "test1"}, {"source": "test2"}],
        ids=["doc1", "doc2"]
    )
    print(f"Collection count after adding: {collection_get.count()}")

    # Reset again to clean up
    print("\n--- Cleaning up ---")
    reset_collection()
    print("Test complete. Vectorstore cleaned up.")
#테스트완료
