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

    def _submit_request(self, payload: Dict[str, Any]) -> FalRequestStatus:
        """Submit a request to the FAL API.

        Args:
            payload (Dict[str, Any]): The request payload

        Returns:
            FalRequestStatus: Status of the submitted request

        Raises:
            RuntimeError: If the API request fails
        """
        url = f"{self.base_url}/{self.model_path}"
        headers = self._get_headers()

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
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
        url = f"{self.base_url}/{self.model_path}/requests/{request_id}/status"
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
        url = f"{self.base_url}/{self.model_path}/requests/{request_id}"
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
        sync_mode: bool = False,
        num_images: int = 1,
        enable_safety_checker: bool = True,
        max_wait_time: int = 60,
        poll_interval: int = 2,
    ) -> Union[FalImageGenerationResponse, FalRequestStatus]:
        """Generate an image using the FAL API.

        Args:
            prompt (str): The prompt to generate an image from
            image_size (ImageSize): Size/aspect ratio of the generated image
            num_inference_steps (int): Number of inference steps
            seed (Optional[int]): Seed for reproducible generation
            guidance_scale (float): How closely to follow the prompt (1-10)
            sync_mode (bool): Whether to wait for the image to be generated
            num_images (int): Number of images to generate
            enable_safety_checker (bool): Whether to enable the safety checker
            max_wait_time (int): Maximum time to wait for image generation in seconds
            poll_interval (int): Time between status checks in seconds

        Returns:
            Union[FalImageGenerationResponse, FalRequestStatus]: The generated image information
                or the request status if sync_mode is False

        Raises:
            RuntimeError: If the API request fails or times out
        """
        # Prepare the payload
        payload = {
            "prompt": prompt,
            "image_size": image_size.value if isinstance(image_size, ImageSize) else image_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
            "sync_mode": sync_mode,
        }

        if seed is not None:
            payload["seed"] = seed

        # For async mode, use the existing _submit_request method to maintain error message
        # consistency
        if not sync_mode:
            request_status = self._submit_request(payload)
            return request_status
        else:
            # For sync mode, submit the request directly
            url = f"{self.base_url}/{self.model_path}"
            headers = self._get_headers()

            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                # Check if the response is already an image generation response
                if "images" in data and "prompt" in data:
                    return FalImageGenerationResponse.model_validate(data)

                # Otherwise, it's a request status that we need to poll
                request_status = FalRequestStatus(
                    request_id=data["request_id"], status=data["status"]
                )

            except requests.exceptions.RequestException as e:
                error_body = (
                    e.response.content.decode()
                    if hasattr(e, "response") and e.response
                    else "No response body"
                )
                raise RuntimeError(
                    f"Failed to generate image: {str(e)}, Response Body: {error_body}"
                )

        # If the request is already completed, return the result
        if request_status.status == "completed":
            return self._get_request_result(request_status.request_id)

        # If the caller doesn't want to wait, return the status
        if not sync_mode:
            return request_status

        # Wait for the request to complete
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            request_status = self._get_request_status(request_status.request_id)
            if request_status.status == "completed":
                return self._get_request_result(request_status.request_id)
            elif request_status.status == "failed":
                raise RuntimeError(f"FAL API request failed: {request_status.request_id}")
            time.sleep(poll_interval)

        # If we get here, the request timed out
        raise RuntimeError(
            f"FAL API request timed out after {max_wait_time} seconds: {request_status.request_id}"
        )
