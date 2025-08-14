import logging

from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import JSONResponse

from crud.firestore import FirestoreCRUD
from crud.langchain import LangChainQA

router = APIRouter()


@router.post("/question", status_code=status.HTTP_200_OK)
@router.post("/question/", include_in_schema=False, status_code=status.HTTP_200_OK)
async def question(
    user: str,
    session: str,
    question: str = Body(..., description="User question for LLM + RAG response"),
):
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        vectors, texts = await FirestoreCRUD.vectors(user=user, session=session)

        response = await LangChainQA.get_answer(
            user_question=question,
            vectors_list=vectors,
            texts_list=texts,
        )

        if not response:
            raise HTTPException(
                status_code=404,
                detail="No relevant information found for the question.",
            )

        return JSONResponse(content={"data": response})

    except Exception as e:
        logging.error(f"Unexpected error occurred: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate response: {str(e)}"
        )
