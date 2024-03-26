from flask import Flask, request, jsonify
from llm.wrapper import setup_qa_chain
from llm.wrapper import query_embeddings
import timeit

app = Flask(__name__)

# Enable debug mode
app.config['DEBUG'] = True

# Enable development mode
app.config['ENV'] = 'development'

@app.route('/ask', methods=['POST'])
def ask():
    """Ask a question about the invoice data and return the answer."""
    question = request.json['question']
    semantic_search = request.json.get('semantic_search', False)
    start = timeit.default_timer()
    if semantic_search:
        semantic_search_results = query_embeddings(question)
        answer = {'semantic_search_results': semantic_search_results}
    else:
        qa_chain = setup_qa_chain()
        response = qa_chain({'query': question})
        answer = {'answer': response['result']}
    end = timeit.default_timer()
    answer['time_taken'] = end - start
    return jsonify(answer)

if __name__ == '__main__':
    # Run the Flask development server on port 3000 with debug mode enabled
    app.run(debug=True, port=3000)

