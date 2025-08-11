"""
Map service for handling map creation and geographic operations.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import folium

# Removed heavy geopandas dependency to simplify deployment on platforms like Railway
import streamlit as st


class MapService:
    """Service for handling map creation and geographic operations."""

    def __init__(self):
        """Initialize the map service."""
        self._boundary_cache = None

    def load_santa_clara_boundary(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Load Santa Clara County boundary from GeoJSON file once per session."""
        if self._boundary_cache is not None:
            return self._boundary_cache

        try:
            # Use pathlib for cross-platform file paths
            script_dir = Path(__file__).parent.parent.parent  # Go up from src/services/
            boundary_file = (
                script_dir / "data" / "boundaries" / "santa_clara_county.geojson"
            )

            # Verify file exists
            if not boundary_file.exists():
                st.warning(f"Boundary file not found at: {boundary_file}")
                self._boundary_cache = (None, None)
                return None, None

            # Load the GeoJSON file without geopandas
            with open(boundary_file, "r") as f:
                boundary_geojson = json.load(f)

            # Extract first feature's properties if available
            properties = {}
            try:
                if (
                    isinstance(boundary_geojson, dict)
                    and boundary_geojson.get("type") == "FeatureCollection"
                    and boundary_geojson.get("features")
                ):
                    properties = (
                        boundary_geojson["features"][0].get("properties", {}) or {}
                    )
            except Exception:
                properties = {}

            self._boundary_cache = (boundary_geojson, properties)
            return boundary_geojson, properties

        except Exception as e:
            st.error(f"Error loading boundary file: {e}")
            self._boundary_cache = (None, None)
            return None, None

    def create_santa_clara_map(self, query_results=None, selected_property=None):
        """Create a Folium map with Santa Clara County boundary and optional query results."""
        # Default Santa Clara County center coordinates
        default_center = [37.3382, -121.8863]
        default_zoom = 10

        # If a property is selected, center on it with higher zoom
        if selected_property and selected_property.get("coordinates"):
            map_center = selected_property["coordinates"]
            map_zoom = 16  # Zoom in closer to the selected property
            m = folium.Map(
                location=map_center, zoom_start=map_zoom, tiles="OpenStreetMap"
            )
        elif query_results and query_results.get("features"):
            # Calculate bounds for all search results and auto-fit
            bounds = self.calculate_bounds(query_results)
            if bounds:
                # Create map without specific center/zoom - will be set by fit_bounds
                m = folium.Map(tiles="OpenStreetMap")
                # Fit map to show all search results
                m.fit_bounds(bounds)
            else:
                # Fallback to default if bounds calculation fails
                m = folium.Map(
                    location=default_center,
                    zoom_start=default_zoom,
                    tiles="OpenStreetMap",
                )
        else:
            # Default map view when no search results
            m = folium.Map(
                location=default_center, zoom_start=default_zoom, tiles="OpenStreetMap"
            )

        # Load boundary from GeoJSON file
        boundary_geojson, properties = self.load_santa_clara_boundary()

        if boundary_geojson:
            # Add the actual county boundary
            folium.GeoJson(
                boundary_geojson,
                style_function=lambda feature: {
                    "fillColor": "transparent",
                    "color": "red",
                    "weight": 2,
                    "fillOpacity": 0.1,
                    "opacity": 0.8,
                },
                highlight=False,  # Disable the blue click highlight box
            ).add_to(m)
        else:
            # Fallback to approximate boundary if file loading fails
            st.warning("Using approximate boundary - GeoJSON file not available")
            santa_clara_bounds = [
                [37.1, -122.2],  # Southwest corner
                [37.1, -121.2],  # Southeast corner
                [37.6, -121.2],  # Northeast corner
                [37.6, -122.2],  # Northwest corner
                [37.1, -122.2],  # Close the polygon
            ]

            folium.Polygon(
                locations=santa_clara_bounds,
                color="red",
                weight=2,
                fill=False,
                popup="Santa Clara County (Approximate Boundary)",
            ).add_to(m)

        # If we have query results, display them prominently
        if query_results and query_results.get("features"):
            # Add query results layer to map
            for idx, feature in enumerate(query_results["features"]):
                # Check if this feature is the selected property
                is_selected = (
                    selected_property and selected_property.get("index") == idx + 1
                )

                # Different styles for selected vs unselected properties
                if is_selected:
                    style = {
                        "fillColor": "red",
                        "color": "darkred",
                        "weight": 3,
                        "fillOpacity": 0.8,
                        "opacity": 1.0,
                    }
                    popup_style = "background-color: red; color: white;"
                    tooltip_style = """
                        background-color: red;
                        color: white;
                        border: 3px solid darkred;
                        border-radius: 3px;
                        box-shadow: 3px;
                    """
                else:
                    style = {
                        "fillColor": "orange",
                        "color": "darkorange",
                        "weight": 2,
                        "fillOpacity": 0.6,
                        "opacity": 1.0,
                    }
                    popup_style = "background-color: orange; color: black;"
                    tooltip_style = """
                        background-color: orange;
                        color: black;
                        border: 2px solid darkorange;
                        border-radius: 3px;
                        box-shadow: 3px;
                    """

                # Add individual feature to map
                folium.GeoJson(
                    {
                        "type": "Feature",
                        "properties": feature["properties"],
                        "geometry": feature["geometry"],
                    },
                    style_function=lambda x, style=style: style,
                    popup=folium.GeoJsonPopup(
                        fields=list(feature["properties"].keys()),
                        localize=True,
                        labels=True,
                        style=popup_style,
                    ),
                    tooltip=folium.GeoJsonTooltip(
                        fields=list(feature["properties"].keys())[:3],
                        localize=True,
                        sticky=True,
                        labels=True,
                        style=tooltip_style,
                        max_width=800,
                    ),
                ).add_to(m)

            st.success(
                f"🎯 Showing {len(query_results['features'])} query result{'s' if len(query_results['features']) != 1 else ''} highlighted in orange"
            )
        else:
            # Show clean map without any default parcel data
            st.info(
                "💬 Use the chat below to search for specific parcels, properties, or areas"
            )

        return m

    @staticmethod
    def extract_center_coordinates(geometry: Dict[str, Any]) -> Optional[List[float]]:
        """Extract center coordinates from geometry for map centering."""
        try:
            if geometry["type"] == "Polygon":
                # Get first ring of polygon
                coords = geometry["coordinates"][0]
                # Calculate centroid (simple average)
                lats = [coord[1] for coord in coords]
                lngs = [coord[0] for coord in coords]
                center_lat = sum(lats) / len(lats)
                center_lng = sum(lngs) / len(lngs)
                return [center_lat, center_lng]
            elif geometry["type"] == "Point":
                return [geometry["coordinates"][1], geometry["coordinates"][0]]
            elif geometry["type"] == "MultiPolygon":
                # Use first polygon
                coords = geometry["coordinates"][0][0]
                lats = [coord[1] for coord in coords]
                lngs = [coord[0] for coord in coords]
                center_lat = sum(lats) / len(lats)
                center_lng = sum(lngs) / len(lngs)
                return [center_lat, center_lng]
        except (KeyError, IndexError, TypeError):
            pass
        return None

    @staticmethod
    def extract_all_coordinates(geometry: Dict[str, Any]) -> List[List[float]]:
        """Extract all coordinates from geometry for bounds calculation."""
        coords = []
        try:
            if geometry["type"] == "Polygon":
                # Get all coordinates from the outer ring
                coords.extend(geometry["coordinates"][0])
            elif geometry["type"] == "Point":
                coords.append(geometry["coordinates"])
            elif geometry["type"] == "MultiPolygon":
                # Get coordinates from all polygons
                for polygon in geometry["coordinates"]:
                    coords.extend(polygon[0])  # Outer ring of each polygon
        except (KeyError, IndexError, TypeError):
            pass
        return coords

    def calculate_bounds(
        self, query_results: Dict[str, Any]
    ) -> Optional[List[List[float]]]:
        """Calculate bounding box for all search result geometries."""
        if not query_results or not query_results.get("features"):
            return None

        all_coords = []

        # Extract coordinates from all features
        for feature in query_results["features"]:
            geometry = feature.get("geometry")
            if geometry:
                coords = self.extract_all_coordinates(geometry)
                all_coords.extend(coords)

        if not all_coords:
            return None

        # Calculate bounds [south, west, north, east]
        lats = [coord[1] for coord in all_coords]
        lngs = [coord[0] for coord in all_coords]

        south = min(lats)
        north = max(lats)
        west = min(lngs)
        east = max(lngs)

        # Add small padding (about 5% of the span)
        lat_padding = (north - south) * 0.05
        lng_padding = (east - west) * 0.05

        return [
            [south - lat_padding, west - lng_padding],  # Southwest corner
            [north + lat_padding, east + lng_padding],  # Northeast corner
        ]
