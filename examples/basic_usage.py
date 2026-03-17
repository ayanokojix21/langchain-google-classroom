"""Example usage of langchain-google-classroom.

Run:

    python examples/basic_usage.py
"""

from langchain_google_classroom import GoogleClassroomLoader


def main() -> None:
    # ----------------------------------------------------------------
    # Example 1: Load all courses with OAuth
    # ----------------------------------------------------------------
    loader = GoogleClassroomLoader(
        client_secrets_file="credentials.json",
        token_file="token.json",
    )
    docs = loader.load()
    if not docs:
        print(  # noqa: T201
            "No documents loaded. Verify credentials, scopes, and course visibility."
        )
        return

    print(f"Loaded {len(docs)} documents")  # noqa: T201

    for doc in docs[:5]:
        print(f"  [{doc.metadata['content_type']}] {doc.metadata['title']}")  # noqa: T201

    # ----------------------------------------------------------------
    # Example 2: Load specific courses, assignments only
    # ----------------------------------------------------------------
    loader = GoogleClassroomLoader(
        course_ids=["123456789"],
        load_assignments=True,
        load_announcements=False,
        load_materials=False,
        service_account_file="service_account.json",
    )
    docs = loader.load()

    # ----------------------------------------------------------------
    # Example 3: Use in a RAG pipeline
    # ----------------------------------------------------------------
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    #
    # splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    # chunks = splitter.split_documents(docs)
    #
    # from langchain_community.vectorstores import FAISS
    # from langchain_openai import OpenAIEmbeddings
    #
    # vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
    # retriever = vectorstore.as_retriever()


if __name__ == "__main__":
    main()
