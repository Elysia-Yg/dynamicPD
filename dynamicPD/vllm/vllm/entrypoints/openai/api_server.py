from http import HTTPStatus

from fastapi import (APIRouter, Depends, HTTPException, Request)
from fastapi.responses import Response
from vllm.entrypoints.openai.protocol import ErrorResponse, validate_json_request, engine_client
from vllm.logger import logger

from dynamicPD.vllm.vllm.entrypoints.openai.protocol import UpdateRequest

router = APIRouter()

@router.get("/stop_profile", response_class=Response)
async def stop_profile(raw_request: Request) -> None:
    try:
        await engine_client(raw_request).stop_profile_npu()
    except Exception as e:
        logger.error(f"Error stopping profile: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            detail=str(e)) from e
    
@router.post("/update_params",
             dependencies=[Depends(validate_json_request)],
              responses={
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.NOT_FOUND.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
             })
async def update_params(request: UpdateRequest, raw_request: Request):

    try:
        await engine_client(raw_request).update_params(request)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            detail=str(e)) from e