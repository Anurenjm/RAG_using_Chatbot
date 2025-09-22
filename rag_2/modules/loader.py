from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader

try:
    from langchain.document_loaders import UnstructuredWordDocumentLoader
except Exception:
    UnstructuredWordDocumentLoader = None

from pathlib import Path

def load_file(filepath):
    """
    Detects file type by extension and returns list[Document]
    Uses LangChain loaders; returns [] if unsupported.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()
    elif suffix in [".txt", ".md"]:
        loader = TextLoader(str(path), encoding="utf8")
        return loader.load()
    elif suffix == ".csv":
        loader = CSVLoader(str(path))
        return loader.load()
    elif suffix in [".docx", ".doc"]:
        if UnstructuredWordDocumentLoader is None:
            raise RuntimeError("UnstructuredWordDocumentLoader not available. Install 'unstructured'.")
        loader = UnstructuredWordDocumentLoader(str(path))
        return loader.load()
    else:
       
        try:
            loader = TextLoader(str(path), encoding="utf8")
            return loader.load()
        except Exception:
            return []
