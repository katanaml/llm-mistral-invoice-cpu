import timeit
import argparse
from llm.wrapper import setup_qa_chain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='What is the invoice number value?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    start = timeit.default_timer()
    qa_chain = setup_qa_chain()
    response = qa_chain({'query': args.input})
    end = timeit.default_timer()

    print(f'\nAnswer: {response["result"]}')
    print('='*50)

    print(f"Time to retrieve answer: {end - start}")
