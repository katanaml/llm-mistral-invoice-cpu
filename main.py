import timeit
import argparse
from llm.wrapper import setup_qa_chain
from llm.wrapper import query_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='What is the invoice number value?',
                        help='Enter the query to pass into the LLM')
    parser.add_argument('--semantic_search',
                        type=bool,
                        default=False,
                        help='Enter True if you want to run semantic search, else False')
    args = parser.parse_args()

    start = timeit.default_timer()
    if args.semantic_search:
        semantic_search = query_embeddings(args.input)
        print(f'Semantic search: {semantic_search}')
        print('='*50)
    else:
        qa_chain = setup_qa_chain()
        response = qa_chain({'query': args.input})
        print(f'\nAnswer: {response["result"]}')
        print('=' * 50)
    end = timeit.default_timer()

    print(f"Time to retrieve answer: {end - start}")
