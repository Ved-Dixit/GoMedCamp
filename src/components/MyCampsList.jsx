// src/components/MyCampsList.jsx
import React from 'react';

const MyCampsList = ({ camps, loading, error, currentUserId, refreshCamps, onCampClick }) => {
  // console.log("MyCampsList: Props received - camps:", camps, "loading:", loading, "error:", error, "onCampClick:", !!onCampClick);

  const handleDeleteCamp = async (campId) => {
    if (!window.confirm("Are you sure you want to delete this camp? This action cannot be undone.")) {
        return;
    }
    // console.log(`MyCampsList: Attempting to delete camp ID: ${campId} by user ID: ${currentUserId}`);
    try {
        const response = await fetch(`https://camp-mdxq.onrender.com/api/organizer/camps/${campId}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
                'X-User-Id': currentUserId.toString(),
            },
        });
        const data = await response.json();
        if (response.ok) {
            alert(data.message || "Camp deleted successfully.");
            if (refreshCamps) {
                // console.log("MyCampsList: Calling refreshCamps after delete.");
                refreshCamps();
            }
        } else {
            alert(data.error || "Failed to delete camp.");
        }
    } catch (err) {
        console.error("Error deleting camp:", err);
        alert("An error occurred while deleting the camp.");
    }
  };

  if (loading) {
    // console.log("MyCampsList: Displaying loading state.");
    return <p>Loading your camps...</p>;
  }

  if (error) {
    // console.log("MyCampsList: Displaying error state:", error);
    // Use a specific class for better styling control if needed
    return <p className="error-message my-camps-error">{error}</p>;
  }

  if (!camps || camps.length === 0) {
    // console.log("MyCampsList: No camps to display or camps array is empty.");
    return <p>You haven't created any camps yet.</p>;
  }

  // console.log(`MyCampsList: Rendering ${camps.length} camps.`);
  return (
    <div className="my-camps-list-container">
      <h4>Your Created Camps</h4>
      <ul className="camps-ul">
        {camps.map(camp => (
          <li key={camp.id} className="camp-list-item">
            <div
              className="camp-item-clickable-content"
              onClick={() => onCampClick && onCampClick(camp.id)}
              role="button"
              tabIndex={0}
              onKeyPress={(e) => { if ((e.key === 'Enter' || e.key === ' ') && onCampClick) onCampClick(camp.id); }}
              aria-label={`View details for camp ${camp.camp_name || camp.name || camp.id}`}
            >
              <div className="camp-item-header">
                <strong>{camp.camp_name || camp.name || `Camp ID: ${camp.id}`}</strong>
                {camp.status && (
                    <span className={`camp-status camp-status-${camp.status.toLowerCase().replace(/\s+/g, '-')}`}>
                        {camp.status}
                    </span>
                )}
              </div>
              <p>
                <strong>Dates:</strong>
                {camp.start_date ? new Date(camp.start_date).toLocaleDateString() : 'N/A'}
                {' to '}
                {camp.end_date ? new Date(camp.end_date).toLocaleDateString() : 'N/A'}
              </p>
              <p>
                <strong>Location:</strong> {camp.location_address || `Lat: ${camp.lat || 'N/A'}, Lng: ${camp.lng || 'N/A'}`}
              </p>
              {camp.description && <p className="camp-description"><strong>Description:</strong> {camp.description}</p>}
            </div>
            <div className="camp-actions">
              <button
                className="action-button delete-button"
                onClick={(e) => {
                  // e.stopPropagation(); // Not strictly necessary here as actions div is separate
                  handleDeleteCamp(camp.id);
                }}
                aria-label={`Delete camp ${camp.camp_name || camp.name || camp.id}`}
              >
                Delete
              </button>
              {/* Example: <button className="action-button edit-button">Edit</button> */}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default MyCampsList;
