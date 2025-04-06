"""
Document upload and processing endpoints for the Local Operator API.

This module contains the FastAPI route handlers for document-related endpoints.
"""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.jobs import JobManager
from local_operator.server.dependencies import (
    get_agent_registry,
    get_config_manager,
    get_credential_manager,
    get_job_manager,
    get_websocket_manager,
)
from local_operator.server.models.schemas import (
    ChatRequest,
    CRUDResponse,
    DocumentProcessRequest,
    DocumentProcessResponse,
    DocumentUploadResponse,
)
from local_operator.server.utils.job_processor_queue import (
    run_job_in_process_with_queue,
)
from local_operator.server.utils.operator import create_operator
from local_operator.server.utils.websocket_manager import WebSocketManager

router = APIRouter(tags=["Documents"])
logger = logging.getLogger("local_operator.server.routes.documents")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path(__file__).parent.parent / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "/v1/documents/upload",
    response_model=CRUDResponse[DocumentUploadResponse],
)
async def upload_document(
    file: UploadFile = File(...),
    credential_manager: CredentialManager = Depends(get_credential_manager),
    config_manager: ConfigManager = Depends(get_config_manager),
) -> JSONResponse:
    """
    Upload a document for processing by the Local Operator.
    
    This endpoint allows users to upload documents (PDF, TXT, DOCX, etc.) that can
    later be processed by the Local Operator.
    
    Args:
        file: The file to upload
        credential_manager: The credential manager instance
        config_manager: The config manager instance
        
    Returns:
        JSONResponse: Response with upload details
    """
    try:
        # Generate a unique filename to prevent collisions
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            file_size = len(content)
        
        # Create a response
        response = DocumentUploadResponse(
            filename=unique_filename,
            size=file_size,
            content_type=file.content_type,
        )
        
        return JSONResponse(
            content={
                "status": 200,
                "message": "Document uploaded successfully",
                "result": response.dict(),
            }
        )
    
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "message": f"Error uploading document: {str(e)}",
            },
        )


@router.post(
    "/v1/documents/process",
    response_model=CRUDResponse[DocumentProcessResponse],
)
async def process_document(
    request: DocumentProcessRequest,
    credential_manager: CredentialManager = Depends(get_credential_manager),
    config_manager: ConfigManager = Depends(get_config_manager),
    job_manager: JobManager = Depends(get_job_manager),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
) -> JSONResponse:
    """
    Process an uploaded document with the Local Operator.
    
    This endpoint processes a previously uploaded document using the specified
    model and hosting service. The document is analyzed according to the provided
    instructions.
    
    Args:
        request: The document processing request
        credential_manager: The credential manager instance
        config_manager: The config manager instance
        job_manager: The job manager instance
        agent_registry: The agent registry instance
        
    Returns:
        JSONResponse: Response with processing task details
    """
    try:
        # Check if the file exists
        file_path = UPLOAD_DIR / request.filename
        if not file_path.exists():
            return JSONResponse(
                status_code=404,
                content={
                    "status": 404,
                    "message": f"Document not found: {request.filename}",
                },
            )
        
        # Create a prompt that includes the document content and instructions
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            document_content = f.read()
        
        prompt = f"""
        # Document Processing Request
        
        ## Instructions
        {request.instructions}
        
        ## Document Content
        ```
        {document_content}
        ```
        
        Please analyze the document according to the instructions above.
        """
        
        # Create a chat request
        chat_request = ChatRequest(
            hosting=request.hosting,
            model=request.model,
            prompt=prompt,
            stream=False,
            options=request.options,
            attachments=[str(file_path)],
        )
        
        # Create an operator
        operator = create_operator(credential_manager, config_manager)
        
        # Process the request
        job_id = job_manager.create_job(
            prompt=chat_request.prompt,
            model=chat_request.model,
            hosting=chat_request.hosting,
        )
        
        # Run the job in a separate process
        await run_job_in_process_with_queue(
            job_id=job_id,
            job_manager=job_manager,
            operator=operator,
            prompt=chat_request.prompt,
            model=chat_request.model,
            hosting=chat_request.hosting,
            stream=chat_request.stream,
            context=None,
            options=chat_request.options,
            attachments=chat_request.attachments,
        )
        
        # Create a response
        response = DocumentProcessResponse(
            task_id=job_id,
            status="processing",
        )
        
        return JSONResponse(
            content={
                "status": 200,
                "message": "Document processing started",
                "result": response.dict(),
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "message": f"Error processing document: {str(e)}",
            },
        )


@router.get(
    "/v1/documents/status/{task_id}",
    response_model=CRUDResponse[DocumentProcessResponse],
)
async def get_document_processing_status(
    task_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> JSONResponse:
    """
    Get the status of a document processing task.
    
    This endpoint returns the current status of a document processing task.
    If the task is complete, it also returns the processing result.
    
    Args:
        task_id: The ID of the document processing task
        job_manager: The job manager instance
        
    Returns:
        JSONResponse: Response with task status and result (if complete)
    """
    try:
        # Get the job from the job manager
        job = job_manager.get_job(task_id)
        if not job:
            return JSONResponse(
                status_code=404,
                content={
                    "status": 404,
                    "message": f"Task not found: {task_id}",
                },
            )
        
        # Create a response
        response = DocumentProcessResponse(
            task_id=task_id,
            status=job.status.value,
            result=job.result.response if job.result else None,
        )
        
        return JSONResponse(
            content={
                "status": 200,
                "message": f"Task status: {job.status.value}",
                "result": response.dict(),
            }
        )
    
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "message": f"Error getting task status: {str(e)}",
            },
        )


@router.get(
    "/v1/documents/download/{filename}",
)
async def download_document(
    filename: str,
) -> FileResponse:
    """
    Download a previously uploaded document.
    
    This endpoint allows users to download a document that was previously uploaded.
    
    Args:
        filename: The name of the file to download
        
    Returns:
        FileResponse: The requested file
    """
    try:
        # Check if the file exists
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Document not found: {filename}")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading document: {str(e)}")


@router.get(
    "/v1/documents/list",
    response_model=CRUDResponse[List[Dict[str, str]]],
)
async def list_documents() -> JSONResponse:
    """
    List all uploaded documents.
    
    This endpoint returns a list of all documents that have been uploaded.
    
    Returns:
        JSONResponse: Response with list of documents
    """
    try:
        # Get all files in the upload directory
        files = []
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                files.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "uploaded_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                })
        
        return JSONResponse(
            content={
                "status": 200,
                "message": "Documents retrieved successfully",
                "result": files,
            }
        )
    
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "message": f"Error listing documents: {str(e)}",
            },
        )


@router.delete(
    "/v1/documents/{filename}",
    response_model=CRUDResponse,
)
async def delete_document(
    filename: str,
) -> JSONResponse:
    """
    Delete a previously uploaded document.
    
    This endpoint allows users to delete a document that was previously uploaded.
    
    Args:
        filename: The name of the file to delete
        
    Returns:
        JSONResponse: Response indicating success or failure
    """
    try:
        # Check if the file exists
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            return JSONResponse(
                status_code=404,
                content={
                    "status": 404,
                    "message": f"Document not found: {filename}",
                },
            )
        
        # Delete the file
        os.remove(file_path)
        
        return JSONResponse(
            content={
                "status": 200,
                "message": f"Document deleted: {filename}",
            }
        )
    
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "message": f"Error deleting document: {str(e)}",
            },
        )