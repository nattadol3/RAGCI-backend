import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient
import os

from get_embedded import *

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context in Thai.:

{context}

---

Answer the question based on the above context: {question}
"""

IMAGE_FOLDER = "chroma"

IMAGE_MAPPING = {
    "ค่าธรรมเนียมการศึกษา": "ค่าธรรมเนียมการศึกษา.png",
    "ค่าเทอม": "ค่าธรรมเนียมการศึกษา.png",
    "ช่องทางติดต่อ": "ช่องทางติดต่อ.png",
    "ค่าธรรมเนียม": "ค่าธรรมเนียมการศึกษา.png",
}

# Get API key from environment variable
api_key = os.environ.get("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("Missing TOGETHER_API_KEY environment variable")

def send_image_response(query_text):
    """Returns a specific image based on the detected keyword in the query."""
    for keyword, image_name in IMAGE_MAPPING.items():
        if keyword in query_text.lower():
            image_path = os.path.join(IMAGE_FOLDER, image_name)
            if os.path.exists(image_path):
                print(f"🖼️ Sending specific image: {image_name}")
                return f"Here is the image related to your query: {image_name}"
    
    return "❌ No matching image found!"

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    image_response = send_image_response(query_text)
    if "❌" not in image_response:
        return image_response  # Return image instead of doing a text search
    
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=25)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    client = InferenceClient(
        provider="together",
        api_key=api_key,
    )

    response_text = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500,
    )
    
    content = response_text.choices[0].message.content

    # Now you can use 'content' as needed
    print("Content = ", content)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"คำตอบ: {response_text}\nSources: {sources}"
    print(formatted_response)
    return content


if __name__ == "__main__":
    main()
