from src.build_index import build_index
from src.evaluate import get_acc
from src.rag_query import cli_rag

def main():
    build_index()
    cli_rag()

if __name__ == '__main__':
    main()
