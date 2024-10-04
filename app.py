import os
from uuid import uuid4

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq.chat_models import ChatGroq
from langchain_community.vectorstores import SupabaseVectorStore

from supabase.client import Client, create_client

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo
from fastapi.responses import JSONResponse

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

question_prompt = """\
You are an expert in Hindu mithology and scriptures. You are well versed with Bhagwat Gita.
Given a question by the user, frame {k} questions that could be answered with reference to Gita.

Follow the output format:
```
{{
  "questions": [
    "question_1",
    "question_2",
    ...
    "question_k"
  ]
}}
```
"""

answer_prompt = """\
You have excess knowledge about hindu mithology and scriptures.
You are given a question from human and context from Bhagwat Gita answer the human's question using the context \
and guide them in the best possible way. Always answer as first person.

Do not answer or cater any inappropriate/NSFW content.

Context from Bhagwat Gita:
{context}
"""

model = ChatGroq(
    name='llama-3.1-70b-versatile',
    api_key=os.environ.get('GROQ_API_KEY')
)

answer_model = ChatGroq(
    name='llama-3.2-90b-text-preview',
    api_key=os.environ.get('GROQ_API_KEY')
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
supabase = create_client(
    os.environ.get('SUPABASE_URL'),
    os.environ.get('SUPABASE_KEY'),
)
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=500,
)

def generate_question(question, k):
    prompts = [
        {"role": "system", "content": question_prompt.format(k=k)},
        {"role": "user", "content": question}
    ]

    response = model.invoke(prompts)
    output_parser = JsonOutputParser()
    output = output_parser.parse(response.content)

    return output["questions"]

def search_book(questions, k):
  results = []
  for question in questions:
    result = vector_store.similarity_search(
      question,
      k=k,
    )

    for res in result:
      results.append(res.page_content)

  return results

def frame_answer(question, context):
  prompts = [
    {"role": "system", "content": answer_prompt.format(context=context)},
    {"role": "user", "content": question}
  ]

  response = answer_model.invoke(prompts)
  return response.content


@app.post("/api/v1/answer")
@limits(calls=10, period=60)
async def answer(request: dict):
    question = request.get("question")
    if not question:
        return JSONResponse({"error": "Missing 'question' in request body"}, status_code=400)

    questions = generate_question(question, 3)
    questions.append(question)

    results = search_book(questions, 2)
    answer = frame_answer(question, '\n\n'.join(results))
    return JSONResponse({
        "answer": answer,
    }, status_code=200)
