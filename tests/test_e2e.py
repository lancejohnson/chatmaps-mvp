import pytest
import re
import os
from typing import List, Optional

from src.services.chat_service import ChatService


def extract_number(text: str) -> Optional[int]:
    """Extract the first number from text, handling various formats."""
    # Remove commas and find numbers
    numbers = re.findall(r"\d{1,3}(?:,\d{3})*", text)
    if numbers:
        return int(numbers[0].replace(",", ""))
    return None


def extract_addresses(text: str) -> List[str]:
    """Extract addresses from text."""
    # Simple pattern for addresses
    addresses = re.findall(
        r"\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Way|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)",
        text,
        re.IGNORECASE,
    )
    return addresses


@pytest.fixture
def chat_service():
    """Fixture to create ChatService for testing."""
    # Ensure we have the required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")

    # Check if database connection is available using PropertyService
    try:
        from src.services.property_service import PropertyService

        property_service = PropertyService()
        if property_service.db_engine is None:
            pytest.fail(
                "Database connection not available - DATABASE_URL not set or connection failed"
            )
    except Exception as e:
        pytest.fail(f"Database connection not available: {e}")

    return ChatService()


class TestPropertyQueries:
    """E2E tests for property-related queries using direct service calls."""

    def test_property_count_over_10_acres(self, chat_service):
        """Test: How many properties are there over 10 acres?"""
        # Process the query directly through the chat service
        result = chat_service.process_chat_with_function_calling(
            "How many properties are there over 10 acres?", []
        )

        # Validate response contains expected number
        expected_number = 6244
        response_text = result["response"]

        # Check if response indicates an error
        if "❌" in response_text or "Error:" in response_text:
            pytest.skip(f"Database or service error: {response_text}")

        extracted_number = extract_number(response_text)

        print(f"DEBUG: Full response: {response_text}")
        print(f"DEBUG: Extracted number: {extracted_number}")

        assert (
            extracted_number is not None
        ), f"Could not extract number from response: {response_text}"
        assert (
            extracted_number == expected_number
        ), f"Expected {expected_number}, got {extracted_number}. Full response: {response_text}"

    def test_find_10_properties_over_10_acres(self, chat_service):
        """Test: Please find 10 properties over 10 acres"""
        result = chat_service.process_chat_with_function_calling(
            "Please find 10 properties over 10 acres", []
        )

        response = result["response"]

        # Check if response indicates an error
        if "❌" in response or "Error:" in response:
            pytest.skip(f"Database or service error: {response}")

        # Should return 10 properties
        assert "10" in response or "ten" in response.lower()

        # Should mention acreage > 10
        assert any(
            phrase in response.lower()
            for phrase in ["10 acres", "over 10", "greater than 10"]
        )

    def test_list_cities_in_santa_clara(self, chat_service):
        """Test: Please list the cities in Santa Clara county"""
        expected_cities = [
            "ALVISO",
            "CAMPBELL",
            "COYOTE",
            "CUPERTINO",
            "GILROY",
            "LOS ALTOS",
            "LOS ALTOS HILLS",
            "LOS GATOS",
            "MILPITAS",
            "MONTE SERENO",
            "MORGAN HILL",
            "MOUNT HAMILTON",
            "MOUNTAIN VIEW",
            "PALO ALTO",
            "REDWOOD ESTATES",
            "SAN JOSE",
            "SAN MARTIN",
            "SANTA CLARA",
            "SARATOGA",
            "STANFORD",
            "SUNNYVALE",
        ]

        result = chat_service.process_chat_with_function_calling(
            "Please list the cities in Santa Clara county", []
        )

        response = result["response"]

        # Check if response indicates an error
        if "❌" in response or "Error:" in response:
            pytest.skip(f"Database or service error: {response}")

        # Check that at least 80% of expected cities are mentioned
        found_cities = []
        for city in expected_cities:
            if city.lower() in response.lower():
                found_cities.append(city)

        coverage = len(found_cities) / len(expected_cities)
        assert (
            coverage >= 0.8
        ), f"Only found {len(found_cities)}/{len(expected_cities)} cities. Found: {found_cities}"

    def test_find_ycombinator_parcel(self, chat_service):
        """Test: Please find that address (YCombinator parcel)"""
        result = chat_service.process_chat_with_function_calling(
            "Please find the address 335 Pioneer Way, Mountain View, CA 94041", []
        )

        response = result["response"]

        # Check if response indicates an error
        if "❌" in response or "Error:" in response:
            pytest.skip(f"Database or service error: {response}")

        # Should return parcel with APN 16066006
        expected_apn = "16066006"
        assert (
            expected_apn in result["search_results"]["features"][0]["properties"]["apn"]
        ), """Expected APN not found in result["search_results"]["features"][0]["properties"]"""

        # Should mention the address
        expected_address = "335 Pioneer Way"
        assert (
            expected_address.lower() in response.lower()
        ), f"Expected address not found in response: {response}"


# Tests can be run with standard pytest commands:
# pytest tests/ -v
# pytest tests/test_e2e.py -v
# pytest tests/test_e2e.py::TestPropertyQueries::test_property_count_over_10_acres -v
