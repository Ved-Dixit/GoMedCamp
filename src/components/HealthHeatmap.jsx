// src/components/HealthHeatmap.jsx
import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet.heat'; // Import the plugin

// Custom component to add heatmap layer as react-leaflet v3+ prefers this for plugins
const HeatmapLayer = ({ data }) => {
  const map = useMap();
  const heatLayerRef = useRef(null);

  useEffect(() => {
    if (heatLayerRef.current) {
      map.removeLayer(heatLayerRef.current); // Remove old layer
    }

    if (data && data.length > 0) {
      // Ensure data points are valid numbers
      const validData = data.filter(point =>
        Array.isArray(point) &&
        point.length === 3 &&
        typeof point[0] === 'number' &&
        typeof point[1] === 'number' &&
        typeof point[2] === 'number' &&
        !isNaN(point[0]) && !isNaN(point[1]) && !isNaN(point[2])
      );

      if (validData.length > 0) {
        heatLayerRef.current = L.heatLayer(validData, {
          radius: 25, // Adjust as needed
          blur: 15,   // Adjust as needed
          maxZoom: 12, // Adjust as needed
          // gradient: {0.4: 'blue', 0.65: 'lime', 1: 'red'} // Optional custom gradient
        }).addTo(map);
      }
    }

    return () => {
      if (heatLayerRef.current) {
        map.removeLayer(heatLayerRef.current);
      }
    };
  }, [data, map]);

  return null; // This component only adds a layer, doesn't render HTML
};


const HealthHeatmap = ({ data }) => {
  // Approximate center of Andhra Pradesh
  const mapCenter = [15.9129, 79.7400];
  const zoomLevel = 7;

  if (!data) {
    return <p>No data available for heatmap.</p>;
  }

  return (
    <MapContainer center={mapCenter} zoom={zoomLevel} style={{ height: '100%', width: '100%' }}>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <HeatmapLayer data={data} />
    </MapContainer>
  );
};

export default HealthHeatmap;
