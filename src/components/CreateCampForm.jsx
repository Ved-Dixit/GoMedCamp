// src/components/CreateCampForm.jsx
import React, { useState } from 'react';

const CreateCampForm = ({ organizerUserId, onCampCreated }) => { // Added onCampCreated prop
  const [campData, setCampData] = useState({
    name: '',
    startDate: '',
    endDate: '',
    location_address: '',
    location_latitude: '',
    location_longitude: '',
    description: '',
    estimatedAttendance: '',
  });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setCampData(prev => ({ ...prev, [name]: value }));
    setError('');
    setSuccess('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    const payload = {
      name: campData.name,
      start_date: campData.startDate,
      end_date: campData.endDate,
      location_address: campData.location_address,
      location_latitude: parseFloat(campData.location_latitude) || null,
      location_longitude: parseFloat(campData.location_longitude) || null,
      description: campData.description,
    };

    if (!payload.name || !payload.start_date || !payload.end_date || payload.location_latitude === null || payload.location_longitude === null) {
        setError('Name, Start Date, End Date, Latitude, and Longitude are required.');
        setLoading(false);
        return;
    }

    try {
      const response = await fetch('https://camp-mdxq.onrender.com/api/organizer/camps', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': organizerUserId.toString(),
        },
        body: JSON.stringify(payload),
      });

      const responseData = await response.json();

      if (response.ok) {
        setSuccess(responseData.message || "Camp created successfully!");
        setCampData({
          name: '', startDate: '', endDate: '',
          location_address: '', location_latitude: '', location_longitude: '',
          description: '', estimatedAttendance: ''
        });
        if (onCampCreated) { // Call the callback to refresh the camp list
          onCampCreated();
        }
      } else {
        setError(responseData.error || `Failed to create camp (Status: ${response.status})`);
      }
    } catch (err) {
      console.error("Create camp error:", err);
      setError("An unexpected error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="create-camp-form-container auth-form-container"> {/* Removed inline styles, use CSS file */}
      <h3>Create New Medical Camp</h3>
      <form onSubmit={handleSubmit} className="auth-form">
        {error && <p className="error-message">{error}</p>}
        {success && <p className="success-message">{success}</p>}

        <div className="form-group">
          <label htmlFor="camp-name">Camp Name:</label>
          <input type="text" id="camp-name" name="name" value={campData.name} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label htmlFor="camp-startDate">Start Date:</label>
          <input type="date" id="camp-startDate" name="startDate" value={campData.startDate} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label htmlFor="camp-endDate">End Date:</label>
          <input type="date" id="camp-endDate" name="endDate" value={campData.endDate} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label htmlFor="camp-location_address">Location Address:</label>
          <input type="text" id="camp-location_address" name="location_address" value={campData.location_address} onChange={handleChange} />
        </div>
        <div className="form-group">
          <label htmlFor="camp-location_latitude">Latitude:</label>
          <input type="number" step="any" id="camp-location_latitude" name="location_latitude" value={campData.location_latitude} onChange={handleChange} required placeholder="e.g., 13.2185"/>
        </div>
        <div className="form-group">
          <label htmlFor="camp-location_longitude">Longitude:</label>
          <input type="number" step="any" id="camp-location_longitude" name="location_longitude" value={campData.location_longitude} onChange={handleChange} required placeholder="e.g., 79.1008"/>
        </div>
        <div className="form-group">
          <label htmlFor="camp-description">Description:</label>
          <textarea id="camp-description" name="description" value={campData.description} onChange={handleChange} />
        </div>
        <div className="form-group">
          <label htmlFor="camp-estimatedAttendance">Estimated Attendance (for your reference):</label>
          <input type="number" id="camp-estimatedAttendance" name="estimatedAttendance" value={campData.estimatedAttendance} onChange={handleChange} />
        </div>
        <button type="submit" disabled={loading} className="auth-button">
          {loading ? 'Creating Camp...' : 'Create Camp'}
        </button>
      </form>
    </div>
  );
};

export default CreateCampForm;
