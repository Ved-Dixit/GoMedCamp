// src/components/OrganizerDashboard.jsx
import React, { useState, useEffect } from 'react';
import CreateCampForm from './CreateCampForm';
import HealthHeatmap from './HealthHeatmap';
import MyCampsList from './MyCampsList';
import { selectableIndicators } from '../data/healthIndicators';
import { indianStatesAndUTs, stateMapCenters } from '../data/IndianStates'; 

const OrganizerDashboard = ({ currentUser, navigateToCampDetails }) => { 
  console.log("OrganizerDashboard: Rendered. currentUser:", JSON.stringify(currentUser));

  const [heatmapData, setHeatmapData] = useState([]);
  const [loadingHeatmap, setLoadingHeatmap] = useState(false);
  const [errorHeatmap, setErrorHeatmap] = useState('');
  const [selectedIndicator, setSelectedIndicator] = useState(selectableIndicators[0].key);
  const [currentIndicatorDetails, setCurrentIndicatorDetails] = useState(selectableIndicators[0]);
  const [currentIndicatorName, setCurrentIndicatorName] = useState('');
  const [selectedMapState, setSelectedMapState] = useState(indianStatesAndUTs.find(s => s.name === "Andhra Pradesh")?.value || indianStatesAndUTs[0].value);
  const [mapView, setMapView] = useState(stateMapCenters[selectedMapState] || stateMapCenters.default);

  const [myCamps, setMyCamps] = useState([]);
  const [loadingCamps, setLoadingCamps] = useState(false);
  const [errorCamps, setErrorCamps] = useState('');

  const fetchMyCamps = async () => {
    if (!currentUser || !currentUser.id) {
      console.warn("fetchMyCamps: Skipping camp fetch. Reason: No currentUser or currentUser.id.", "currentUser:", JSON.stringify(currentUser));
      setMyCamps([]); 
      setLoadingCamps(false); 
      setErrorCamps(''); 
      return;
    }

    console.log(`fetchMyCamps: Attempting to fetch camps for user ID: ${currentUser.id}`);
    setLoadingCamps(true);
    setErrorCamps('');
    try {
      const response = await fetch('https://camp-mdxq.onrender.com/api/organizer/camps', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': currentUser.id.toString(), 
        },
      });

      const responseText = await response.text(); 
      if (!response.ok) {
        let errorDetails = `Failed to fetch camps (Status: ${response.status} ${response.statusText})`;
        try {
          const errorJson = JSON.parse(responseText); 
          errorDetails = errorJson.error || errorDetails;
        } catch (e) {
          errorDetails = responseText || errorDetails;
        }
        console.error("fetchMyCamps: API error details:", errorDetails);
        throw new Error(errorDetails);
      }

      const campsData = JSON.parse(responseText); 
      console.log("fetchMyCamps: Successfully received and parsed camps data:", campsData);
      const processedCamps = campsData.map(camp => ({ ...camp, id: camp.camp_id || camp.id }));
      setMyCamps(processedCamps); 

    } catch (err) {
      console.error("Error in fetchMyCamps catch block:", err);
      setErrorCamps(err.message || "Could not load your camps due to an unexpected error.");
      setMyCamps([]); 
    } finally {
      setLoadingCamps(false);
    }
  };

  useEffect(() => {
    console.log("OrganizerDashboard: useEffect for currentUser triggered. CurrentUser ID:", currentUser?.id);
    if (currentUser && currentUser.id) {
      fetchMyCamps(); 
    } else {
      console.log("OrganizerDashboard: useEffect for currentUser - no valid user, clearing camps.");
      setMyCamps([]);
      setLoadingCamps(false);
      setErrorCamps('');
    }
  }, [currentUser]); 

  useEffect(() => {
    const fetchHeatmapData = async () => {
      if (!selectedIndicator || !selectedMapState) return;
      setLoadingHeatmap(true);
      setErrorHeatmap('');
      setCurrentIndicatorName('');
      setHeatmapData([]);

      // Add a 50-second delay
      console.log("OrganizerDashboard: Starting 50-second delay for heatmap data fetch.");
      await new Promise(resolve => setTimeout(resolve, 50000)); // 50000 milliseconds = 50 seconds
      console.log("OrganizerDashboard: Delay finished. Fetching heatmap data.");

      try {
        const response = await fetch(`https://camp-mdxq.onrender.com/api/heatmap_data?state=${selectedMapState}&indicator_id=${selectedIndicator}`);
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || `Failed to fetch heatmap data (Status: ${response.status})`);
        }
        const geoJsonData = await response.json();
        const indicatorDetail = selectableIndicators.find(ind => ind.key === selectedIndicator);
        setCurrentIndicatorDetails(indicatorDetail);
        if (geoJsonData.metadata && geoJsonData.metadata.full_indicator_name) {
            setCurrentIndicatorName(geoJsonData.metadata.full_indicator_name);
        } else if (indicatorDetail) {
            setCurrentIndicatorName(indicatorDetail.label);
        }
        if (geoJsonData && geoJsonData.features && geoJsonData.features.length > 0) {
          const processedData = geoJsonData.features
            .filter(feature =>
              feature.geometry && feature.geometry.coordinates &&
              feature.properties && feature.properties.value != null &&
              !isNaN(parseFloat(feature.properties.value))
            )
            .map(feature => {
              let intensity = parseFloat(feature.properties.value);
              if (indicatorDetail && indicatorDetail.invert) {
                intensity = 100 - intensity;
                intensity = Math.max(0, intensity);
              }
              return [feature.geometry.coordinates[1], feature.geometry.coordinates[0], intensity];
            });
          setHeatmapData(processedData);
        } else {
          setHeatmapData([]);
          console.log("No features found in heatmap data for the selected criteria.");
        }
      } catch (err) {
        console.error("Error fetching heatmap data:", err);
        setErrorHeatmap(err.message || "Could not load heatmap data.");
        setHeatmapData([]);
      } finally {
        setLoadingHeatmap(false);
      }
    };
    fetchHeatmapData();
  }, [selectedIndicator, selectedMapState]);

  const handleIndicatorChange = (e) => setSelectedIndicator(e.target.value);
  const handleMapStateChange = (e) => {
    const newStateValue = e.target.value;
    setSelectedMapState(newStateValue);
    setMapView(stateMapCenters[newStateValue] || stateMapCenters.default);
  };


  if (!currentUser || currentUser.userType !== 'organizer') {
    console.warn("OrganizerDashboard: Access Denied render. currentUser:", JSON.stringify(currentUser));
    return <p>Access Denied. This dashboard is for Camp Organizers only.</p>;
  }

  const selectedStateDisplayName = indianStatesAndUTs.find(s => s.value === selectedMapState)?.name || selectedMapState;
  console.log("OrganizerDashboard: Props being passed to MyCampsList -> camps:", JSON.stringify(myCamps), "loading:", loadingCamps, "error:", errorCamps);

  return (
    <div className="organizer-dashboard">
      <div className="dashboard-left-panel">
        <CreateCampForm organizerUserId={currentUser.id} onCampCreated={fetchMyCamps} />
        <hr className="panel-divider" />
        <MyCampsList
            camps={myCamps}
            loading={loadingCamps}
            error={errorCamps}
            currentUserId={currentUser.id}
            refreshCamps={fetchMyCamps}
            onCampClick={navigateToCampDetails} // Pass the navigation function here
        />
      </div>
      <div className="dashboard-right-panel">
        <h2>Regional Health Indicators Heatmap ({selectedStateDisplayName})</h2>
        <div className="heatmap-controls">
          <div>
            <label htmlFor="state-select">Select State/UT: </label>
            <select id="state-select" value={selectedMapState} onChange={handleMapStateChange}>
              {indianStatesAndUTs.map(state => (
                <option key={state.value} value={state.value}>{state.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="indicator-select">Select Indicator: </label>
            <select id="indicator-select" value={selectedIndicator} onChange={handleIndicatorChange}>
              {selectableIndicators.map(ind => (
                <option key={ind.key} value={ind.key}>{ind.label}</option>
              ))}
            </select>
          </div>
        </div>
        {currentIndicatorName && (
          <p className="indicator-info">
            Displaying: <strong>{currentIndicatorName}</strong>
            {currentIndicatorDetails && currentIndicatorDetails.invert && " (Higher intensity means lower original value - indicating higher need)"}
          </p>
        )}
        {loadingHeatmap && <p>Loading heatmap...</p>}
        {errorHeatmap && <p className="error-message">{errorHeatmap}</p>}
        {!loadingHeatmap && !errorHeatmap && heatmapData.length > 0 && (
          <HealthHeatmap data={heatmapData} mapCenter={mapView.center} mapZoom={mapView.zoom} key={selectedMapState + selectedIndicator} />
        )}
        {!loadingHeatmap && !errorHeatmap && heatmapData.length === 0 && (
          <p>No data points to display on the map for the selected state and indicator.</p>
        )}
      </div>
    </div>
  );
};

export default OrganizerDashboard;
