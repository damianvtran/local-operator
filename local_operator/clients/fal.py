"""
The FAL client for Local Operator.

This module provides a client for interacting with the FAL API to generate images
using the FLUX.1 text-to-image model.
"""

import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import requests
from pydantic import BaseModel, SecretStr


class ImageSize(str, Enum):
    """Image size options for the FAL API."""

    SQUARE_HD = "square_hd"
    SQUARE = "square"
    PORTRAIT_4_3 = "portrait_4_3"
    PORTRAIT_16_9 = "portrait_16_9"
    LANDSCAPE_4_3 = "landscape_4_3"
    LANDSCAPE_16_9 = "landscape_16_9"


class GenerationType(str, Enum):
    """Generation type options for the FAL API."""

    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"


class FalImage(BaseModel):
    """Image information returned by the FAL API.

    Attributes:
        url (str): URL of the generated image
        width (Optional[int]): Width of the image in pixels
        height (Optional[int]): Height of the image in pixels
        content_type (str): Content type of the image (e.g., "image/jpeg")
    """

    url: str
    width: Optional[int] = None
    height: Optional[int] = None
    content_type: str = "image/jpeg"


class FalImageGenerationResponse(BaseModel):
    """Response from the FAL API for image generation.

    Attributes:
        images (List[FalImage]): List of generated images
        prompt (str): The prompt used for generating the image
        seed (Optional[int]): Seed used for generation
        has_nsfw_concepts (Optional[List[bool]]): Whether the images contain NSFW concepts
    """

    images: List[FalImage]
    prompt: str
    seed: Optional[int] = None
    has_nsfw_concepts: Optional[List[bool]] = None


class FalRequestStatus(BaseModel):
    """Status of a FAL API request.

    Attributes:
        request_id (str): ID of the request
        status (str): Status of the request (e.g., "completed", "processing")
    """

    request_id: str
    status: str


class FalClient:
    """Client for interacting with the FAL API.

    This client is used to generate images using the FLUX.1 text-to-image model.
    """

    def __init__(self, api_key: SecretStr, base_url: str = "https://queue.fal.run") -> None:
        """Initialize the FalClient.

        Args:
            api_key (SecretStr): The FAL API key
            base_url (str): The base URL for the FAL API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_path = "fal-ai/flux/dev"

        if not self.api_key:
            raise ValueError("FAL API key is required")

    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for the FAL API request.

        Returns:
            Dict[str, str]: Headers for the API request
        """
        return {
            "Authorization": f"Key {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def _submit_request(
        self,
        payload: Dict[str, Any],
        generation_type: GenerationType = GenerationType.TEXT_TO_IMAGE,
    ) -> Union[FalRequestStatus, Dict[str, Any]]:
        """Submit a request to the FAL API.

        Args:
            payload (Dict[str, Any]): The request payload
            generation_type (GenerationType): The type of generation to perform

        Returns:
            Union[FalRequestStatus, Dict[str, Any]]: Status of the submitted request or
            direct response for sync mode

        Raises:
            RuntimeError: If the API request fails
        """
        # For text-to-image, use the base URL
        # For image-to-image, append the endpoint to the URL
        if generation_type == GenerationType.TEXT_TO_IMAGE:
            url = f"{self.base_url}/{self.model_path}"
        else:
            url = f"{self.base_url}/{self.model_path}/{generation_type.value}"

        headers = self._get_headers()

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Handle sync mode response which directly returns the image generation result
            # without a request_id or status
            if payload.get("sync_mode", False) and "images" in data:
                # This is a direct image generation response, not a request status
                return data

            # Regular async mode response with request_id and status
            return FalRequestStatus(request_id=data["request_id"], status=data["status"])
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to submit FAL API request: {str(e)}, Response Body: {error_body}"
            )

    def _get_request_status(self, request_id: str) -> FalRequestStatus:
        """Get the status of a FAL API request.

        Args:
            request_id (str): ID of the request

        Returns:
            FalRequestStatus: Status of the request

        Raises:
            RuntimeError: If the API request fails
        """
        # The correct URL format for status checks
        url = f"{self.base_url}/fal-ai/flux/requests/{request_id}/status"
        headers = self._get_headers()

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            return FalRequestStatus(request_id=request_id, status=data["status"])
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to get FAL API request status: {str(e)}, Response Body: {error_body}"
            )

    def _get_request_result(self, request_id: str) -> FalImageGenerationResponse:
        """Get the result of a completed FAL API request.

        Args:
            request_id (str): ID of the request

        Returns:
            FalImageGenerationResponse: The generated image information

        Raises:
            RuntimeError: If the API request fails
        """
        # The correct URL format for result retrieval
        url = f"{self.base_url}/fal-ai/flux/requests/{request_id}"
        headers = self._get_headers()

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            return FalImageGenerationResponse.model_validate(data)
        except requests.exceptions.RequestException as e:
            error_body = (
                e.response.content.decode()
                if hasattr(e, "response") and e.response
                else "No response body"
            )
            raise RuntimeError(
                f"Failed to get FAL API request result: {str(e)}, Response Body: {error_body}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to parse FAL API response: {str(e)}")

    def generate_image(
        self,
        prompt: str,
        image_size: ImageSize = ImageSize.LANDSCAPE_4_3,
        num_inference_steps: int = 28,
        seed: Optional[int] = None,
        guidance_scale: float = 3.5,
        sync_mode: bool = True,  # This parameter is passed to the FAL API
        num_images: int = 1,
        enable_safety_checker: bool = True,
        max_wait_time: int = 60,
        poll_interval: int = 2,
        image_url: Optional[str] = None,
        strength: Optional[float] = None,
    ) -> FalImageGenerationResponse:
        """Generate an image using the FAL API.

        Args:
            prompt (str): The prompt to generate an image from
            image_size (ImageSize): Size/aspect ratio of the generated image
            num_inference_steps (int): Number of inference steps
            seed (Optional[int]): Seed for reproducible generation
            guidance_scale (float): How closely to follow the prompt (1-10)
            sync_mode (bool): Whether to use sync_mode in the FAL API request.
                This affects how the FAL API handles the request but our function
                will always wait for the result.
            num_images (int): Number of images to generate
            enable_safety_checker (bool): Whether to enable the safety checker
            max_wait_time (int): Maximum time to wait for image generation in seconds
            poll_interval (int): Time between status checks in seconds
            image_url (Optional[str]): URL of the image to use as a base for
                image-to-image generation
            strength (Optional[float]): Strength parameter for image-to-image generation (0-1)

        Returns:
            FalImageGenerationResponse: The generated image information

        Raises:
            RuntimeError: If the API request fails or times out
        """
        # Determine generation type based on parameters
        if image_url is not None:
            generation_type = GenerationType.IMAGE_TO_IMAGE
        else:
            generation_type = GenerationType.TEXT_TO_IMAGE

        # Prepare the payload
        payload = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
            "sync_mode": sync_mode,  # Include sync_mode in the payload
        }

        # Add parameters specific to the generation type
        if generation_type == GenerationType.TEXT_TO_IMAGE:
            payload["image_size"] = (
                image_size.value if isinstance(image_size, ImageSize) else image_size
            )
        else:  # IMAGE_TO_IMAGE
            payload["image_url"] = image_url
            if strength is not None:
                payload["strength"] = strength

        if seed is not None:
            payload["seed"] = seed

        # Submit the request using the _submit_request method
        response = self._submit_request(payload, generation_type)

        # Handle direct response from sync mode
        if isinstance(response, dict) and "images" in response:
            return FalImageGenerationResponse.model_validate(response)

        # Handle async response with request status
        request_status = response if isinstance(response, FalRequestStatus) else None

        # If the request is already completed, return the result
        if request_status and request_status.status == "completed":
            return self._get_request_result(request_status.request_id)

        # Poll for the result
        start_time = time.time()
        # Ensure request_status is not None before proceeding
        if request_status is None:
            raise RuntimeError("Failed to get request status from FAL API")

        request_id = request_status.request_id

        while time.time() - start_time < max_wait_time:
            try:
                request_status = self._get_request_status(request_id)

                # Check status and take appropriate action
                if request_status.status.upper() == "COMPLETED":
                    return self._get_request_result(request_id)
                elif request_status.status.upper() == "FAILED":
                    raise RuntimeError(f"FAL API request failed: {request_id}")

                # Wait before polling again
                time.sleep(poll_interval)
            except Exception:
                # Continue polling
                time.sleep(poll_interval)

                # If we've spent more than half the max wait time with errors, try to get the result
                if time.time() - start_time > max_wait_time / 2:
                    try:
                        # Sometimes the status endpoint might fail but the result is ready
                        return self._get_request_result(request_id)
                    except Exception:
                        # If that fails too, continue polling
                        pass

        # If we get here, the request timed out
        # Try one last time to get the result directly before giving up
        try:
            return self._get_request_result(request_id)
        except Exception:
            raise RuntimeError(
                f"FAL API request timed out after {max_wait_time} seconds: {request_id}"
            )
