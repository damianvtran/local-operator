from typing import Any, Dict
from urllib.parse import urlencode

import requests
from pydantic import BaseModel, Field, SecretStr


class SerpApiSearchMetadata(BaseModel):
    """Metadata about a SERP API search request.

    Attributes:
        id (str): Unique identifier for the search request
        status (str): Status of the request (e.g. "Success")
        json_endpoint (str): URL to fetch JSON results
        created_at (str): Timestamp when request was created
        processed_at (str): Timestamp when request was processed
        google_url (str): Original Google search URL
        raw_html_file (str): URL to raw HTML results
        total_time_taken (float): Total processing time in seconds
    """

    id: str
    status: str
    json_endpoint: str
    created_at: str
    processed_at: str
    google_url: str
    raw_html_file: str
    total_time_taken: float
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable.

        Returns:
            Dict[str, Any]: A JSON serializable dictionary representation of the model.
        """
        return super().model_dump(*args, **kwargs)


class SerpApiSearchParameters(BaseModel):
    """Parameters used for a SERP API search request.

    Attributes:
        engine (str): Search engine used (e.g. "google")
        q (str): Search query string
        location_requested (str | None): Location that was requested
        location_used (str | None): Location that was actually used
        google_domain (str): Google domain used (e.g. "google.com")
        hl (str): Language code
        gl (str): Country code
        device (str): Device type used
    """

    engine: str
    q: str
    location_requested: str | None = Field(default=None)
    location_used: str | None = Field(default=None)
    google_domain: str = Field(default="google.com")
    hl: str = Field(default="en")
    gl: str = Field(default="us")
    device: str
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable.

        Returns:
            Dict[str, Any]: A JSON serializable dictionary representation of the model.
        """
        return super().model_dump(*args, **kwargs)


class SerpApiSearchInformation(BaseModel):
    """Information about the search results.

    Attributes:
        organic_results_state (str): State of organic results
        query_displayed (str): Query that was displayed
        total_results (int): Total number of results found
        time_taken_displayed (float): Time taken to display results
    """

    organic_results_state: str
    query_displayed: str
    total_results: int | None = None
    time_taken_displayed: float | None = None
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiRecipeResult(BaseModel):
    """Recipe result from SERP API search.

    Attributes:
        title (str): Recipe title
        link (str): URL to recipe
        source (str): Source website/author
        total_time (str | None): Total recipe time
        ingredients (list[str]): List of ingredients
        thumbnail (str): URL to recipe thumbnail image
        rating (float | None): Recipe rating
        reviews (int | None): Number of reviews
    """

    title: str
    link: str
    source: str
    total_time: str | None = None
    ingredients: list[str]
    thumbnail: str
    rating: float | None = None
    reviews: int | None = None
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiShoppingResult(BaseModel):
    """Shopping result from SERP API search.

    Attributes:
        position (int): Position in results
        block_position (str): Block position in results
        title (str): Product title
        price (str): Displayed price
        extracted_price (float): Numeric price value
        link (str): Product URL
        source (str): Seller/source
        reviews (int | None): Number of reviews
        thumbnail (str): Product thumbnail URL
    """

    position: int
    block_position: str
    title: str
    price: str
    extracted_price: float
    link: str
    source: str
    reviews: int | None = None
    thumbnail: str
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiLocalResult(BaseModel):
    """Local business result from SERP API search.

    Attributes:
        position (int): Position in results
        title (str): Business name
        place_id (str): Google Place ID
        lsig (str): Location signature
        place_id_search (str): URL to place search
        rating (float | None): Business rating
        reviews (int | None): Number of reviews
        price (str | None): Price level indicator
        type (str): Business type/category
        address (str): Business address
        thumbnail (str): Business thumbnail image URL
        gps_coordinates (dict): Latitude and longitude coordinates
    """

    position: int
    title: str
    place_id: str
    lsig: str
    place_id_search: str
    rating: float | None = None
    reviews: int | None = None
    price: str | None = None
    type: str
    address: str
    thumbnail: str
    gps_coordinates: Dict[str, float]
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiSiteLink(BaseModel):
    """Inline sitelink in organic search results.

    Attributes:
        title (str): Title of the linked page
        link (str): URL of the linked page
    """

    title: str
    link: str
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiRichSnippetExtensions(BaseModel):
    """Detected extensions in rich snippets.

    Attributes:
        introduced_th_century (int | None): Century when item was introduced
    """

    introduced_th_century: int | None = None
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiRichSnippet(BaseModel):
    """Rich snippet information for organic results.

    Attributes:
        bottom (dict[str, Any] | None): Bottom snippet information including extensions
    """

    bottom: Dict[str, Any] | None = None
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiResultSource(BaseModel):
    """Source information for organic results.

    Attributes:
        description (str): Description of the source website
        source_info_link (str): Link to more information about the source
        security (str): Security status of the source
        icon (str): URL of the source icon
    """

    description: str
    source_info_link: str
    security: str
    icon: str
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiAboutResult(BaseModel):
    """Additional information about organic results.

    Attributes:
        source (SerpApiResultSource): Information about the result source
        keywords (list[str]): Keywords associated with the result
        languages (list[str]): Languages the result is available in
        regions (list[str]): Geographic regions the result is relevant to
    """

    source: SerpApiResultSource
    keywords: list[str]
    languages: list[str]
    regions: list[str]
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiOrganicResult(BaseModel):
    """Organic search result from SERP API.

    Attributes:
        position (int): Position in search results
        title (str): Title of the result
        link (str): URL of the result
        redirect_link (str | None): Google redirect URL
        displayed_link (str): URL shown in search results
        snippet (str): Text snippet from the result
        sitelinks (dict[str, list[SerpApiSiteLink]] | None): Related page links
        rich_snippet (SerpApiRichSnippet | None): Enhanced result information
        about_this_result (SerpApiAboutResult | None): Additional result metadata
        about_page_link (str | None): Link to Google's about page
        about_page_serpapi_link (str | None): SerpAPI link for about page
        cached_page_link (str | None): Link to cached version
        related_pages_link (str | None): Link to related pages
    """

    position: int
    title: str
    link: str
    redirect_link: str | None = None
    displayed_link: str
    snippet: str
    sitelinks: Dict[str, list[SerpApiSiteLink]] | None = None
    rich_snippet: SerpApiRichSnippet | None = None
    about_this_result: SerpApiAboutResult | None = None
    about_page_link: str | None = None
    about_page_serpapi_link: str | None = None
    cached_page_link: str | None = None
    related_pages_link: str | None = None
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiGpsCoordinates(BaseModel):
    """GPS coordinates for a local business result.

    Attributes:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
    """

    latitude: float
    longitude: float
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiLocalPlace(BaseModel):
    """Local business result from SERP API.

    Attributes:
        position (int): Position in search results
        title (str): Name of the business
        place_id (str): Google Place ID
        lsig (str): Google signature
        place_id_search (str): URL to search this place on SERP API
        rating (float | None): Business rating out of 5
        reviews (int | None): Number of reviews
        price (str | None): Price level indicator ($, $$, etc)
        type (str): Type of business
        address (str): Business address
        thumbnail (str): URL to business image thumbnail
        gps_coordinates (SerpApiGpsCoordinates): Business location coordinates
    """

    position: int
    title: str
    place_id: str
    lsig: str
    place_id_search: str
    rating: float | None = None
    reviews: int | None = None
    price: str | None = None
    type: str
    address: str
    thumbnail: str
    gps_coordinates: SerpApiGpsCoordinates
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiLocalResults(BaseModel):
    """Local business results section from SERP API.

    Attributes:
        more_locations_link (str): URL to view more local results
        places (list[SerpApiLocalPlace]): List of local business results
    """

    more_locations_link: str
    places: list[SerpApiLocalPlace]
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiResponse(BaseModel):
    """Complete response from SERP API search.

    Attributes:
        search_metadata (SerpApiSearchMetadata): Search request metadata
        search_parameters (SerpApiSearchParameters): Search request parameters
        search_information (SerpApiSearchInformation): Search results information
        recipes_results (list[SerpApiRecipeResult] | None): Recipe results if any
        shopping_results (list[SerpApiShoppingResult] | None): Shopping results if any
        local_results (SerpApiLocalResults | None): Local business results if any
        organic_results (list[SerpApiOrganicResult] | None): Organic search results
        related_searches (list[Dict[str, Any]] | None): Related search queries
        pagination (Dict[str, Any] | None): Pagination information
    """

    search_metadata: SerpApiSearchMetadata
    search_parameters: SerpApiSearchParameters
    search_information: SerpApiSearchInformation
    recipes_results: list[SerpApiRecipeResult] | None = None
    shopping_results: list[SerpApiShoppingResult] | None = None
    local_results: SerpApiLocalResults | None = None
    organic_results: list[SerpApiOrganicResult] | None = None
    related_searches: list[Dict[str, Any]] | None = None
    pagination: Dict[str, Any] | None = None
    # Allow additional fields
    model_config = {"extra": "allow"}

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary, making it JSON serializable."""
        return super().model_dump(*args, **kwargs)


class SerpApiClient:
    """Client for making requests to the SERP API.

    Attributes:
        api_key (str): SERP API key for authentication
    """

    def __init__(self, api_key: SecretStr):
        """Initialize the SERP API client.

        Args:
            api_key (str | None): SERP API key. If not provided, will \
          try to get from SERP_API_KEY env var.

        Raises:
            RuntimeError: If no API key is provided or found in environment.
        """
        self.api_key = api_key
        if not self.api_key:
            raise RuntimeError(
                "SERP API key must be provided or set in SERP_API_KEY environment variable"
            )

    def search(
        self,
        query: str,
        engine: str = "google",
        num_results: int = 20,
        location: str | None = None,
        language: str | None = None,
        country: str | None = None,
        device: str = "desktop",
    ) -> SerpApiResponse:
        """Execute a search using the SERP API.

        Makes an HTTP request to SERP API with the provided parameters and
        returns a structured response.

        Args:
            query (str): The search query string
            engine (str, optional): Search engine to use (default: "google")
            num_results (int, optional): Number of results to return (default: 20)
            location (str | None, optional): Geographic location to search from
            language (str | None, optional): Language code for results (e.g. "en")
            country (str | None, optional): Country code for results (e.g. "us")
            device (str, optional): Device type to emulate (default: "desktop")

        Returns:
            SerpApiResponse containing structured search results

        Raises:
            RuntimeError: If the API request fails
        """
        # Build query parameters
        params = {
            "api_key": self.api_key.get_secret_value(),
            "q": query,
            "engine": engine,
            "num": num_results,
            "device": device,
        }
        if location:
            params["location"] = location
        if language:
            params["hl"] = language
        if country:
            params["gl"] = country

        url = f"https://serpapi.com/search?{urlencode(params)}"
        try:
            response = requests.get(url)
            if response.status_code != 200:
                raise RuntimeError(
                    f"SERP API request failed with status {response.status_code}, content:"
                    f" {response.content.decode()}"
                )
            data = response.json()
            return SerpApiResponse.model_validate(data)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to execute SERP API search due to a requests error: {str(e)}, Response"
                f" Body: {e.response.content.decode() if e.response else 'No response body'}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to execute SERP API search: {str(e)}") from e
