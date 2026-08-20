import os
from groq import Groq
from duckduckgo_search import DDGS


client = Groq(
    api_key="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"
)



def web_search(question):

    results = []

    try:

        with DDGS() as ddgs:

            search_results = ddgs.text(
                question,
                max_results=3
            )


            for result in search_results:

                results.append(
                    result["body"]
                )


        return "\n\n".join(results)


    except Exception as e:

        return f"Web search failed: {e}"





def generate_answer(question, context, source):


    prompt = f"""

You are LuminaPDF AI.

Answer the question using the information below.

Source:
{source}


Information:
{context}


Question:
{question}


Answer:
"""


    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )


    return response.choices[0].message.content





def ask_gemini(question, context):


    pdf_prompt = f"""

You are a PDF assistant.

Answer ONLY from the PDF context.

If the answer is not present,
reply exactly:
NOT_FOUND


PDF Context:

{context}


Question:

{question}


Answer:
"""


    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role":"user",
                "content":pdf_prompt
            }
        ]
    )


    answer = response.choices[0].message.content



    # PDF does not contain answer
    if "NOT_FOUND" in answer:


        web_context = web_search(question)


        return generate_answer(
            question,
            web_context,
            "Web Search"
        )


    return answer