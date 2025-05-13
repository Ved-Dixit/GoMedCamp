// src/components/ManageFollowUpsPage.jsx
import React, { useState, useEffect, useCallback } from 'react';
import api from '../services/api';
import '../App.css';

const ManageFollowUpsPage = ({ campId, currentUser, setCurrentPage }) => {
  const [followUpPatients, setFollowUpPatients] = useState([]);
  const [campName, setCampName] = useState('');
  const [newPatientIdentifier, setNewPatientIdentifier] = useState('');
  const [newPatientNotes, setNewPatientNotes] = useState('');
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const fetchCampDetailsAndFollowUps = useCallback(async () => {
    if (!campId) {
      setError("Camp ID is missing.");
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const campDetailsRes = await api.get(`/organizer/camps/${campId}`);
      setCampName(campDetailsRes.data.name || `Camp ID ${campId}`);

      const followUpsRes = await api.get(`/camps/${campId}/patients/followup`);
      setFollowUpPatients(followUpsRes.data || []);
      setError('');
    } catch (err) {
      setError('Failed to load follow-up data. ' + (err.response?.data?.error || err.message));
      console.error("Fetch follow-up data error:", err);
    } finally {
      setLoading(false);
    }
  }, [campId]);

  useEffect(() => {
    if (!currentUser || currentUser.userType !== 'organizer') {
      setCurrentPage('login');
      return;
    }
    fetchCampDetailsAndFollowUps();
  }, [currentUser, campId, setCurrentPage, fetchCampDetailsAndFollowUps]);

  const handleAddPatientForFollowUp = async (e) => {
    e.preventDefault();
    if (!newPatientIdentifier.trim()) {
      setError('Patient identifier is required.');
      return;
    }
    setError('');
    setSuccess('');
    try {
      await api.post(`/camps/${campId}/patients/followup`, {
        patientIdentifier: newPatientIdentifier.trim(),
        notes: newPatientNotes.trim(),
      });
      setSuccess('Patient added for follow-up successfully!');
      setNewPatientIdentifier('');
      setNewPatientNotes('');
      fetchCampDetailsAndFollowUps(); // Refresh list
    } catch (err) {
      setError('Failed to add patient for follow-up. ' + (err.response?.data?.error || err.message));
      console.error("Add follow-up error:", err);
    }
  };

  if (loading) return <div className="page-loading-message">Loading follow-up management...</div>;
  
  return (
    <div className="container-page manage-follow-ups-page">
      <button
        onClick={() => setCurrentPage('campDetails')}
        className="back-button"
        style={{ marginBottom: '20px' }}
      >
        &larr; Back to Camp Details
      </button>
      <div className="page-header">
        <h1>Manage Patient Follow-ups for {campName}</h1>
      </div>

      {error && <div className="error-message">{error}</div>}
      {success && <div className="success-message">{success}</div>}

      <div className="page-section add-followup-section">
        <h2>Add Patient for Follow-up</h2>
        <form onSubmit={handleAddPatientForFollowUp} className="add-followup-form">
          <div className="form-group">
            <label htmlFor="patient-identifier">Patient Identifier (e.g., Email, Phone, Unique ID):</label>
            <input
              type="text"
              id="patient-identifier"
              value={newPatientIdentifier}
              onChange={(e) => setNewPatientIdentifier(e.target.value)}
              placeholder="Enter patient's unique identifier"
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="followup-notes">Notes (optional):</label>
            <textarea
              id="followup-notes"
              value={newPatientNotes}
              onChange={(e) => setNewPatientNotes(e.target.value)}
              placeholder="Any specific notes for follow-up"
              rows="3"
            />
          </div>
          <button type="submit" className="button-primary">Add Patient</button>
        </form>
      </div>

      <div className="page-section existing-followups-section">
        <h2>Patients Scheduled for Follow-up</h2>
        {followUpPatients.length > 0 ? (
          <ul className="followup-patients-list">
            {followUpPatients.map(patient => (
              <li key={patient.id} className="followup-patient-item">
                <p><strong>Identifier:</strong> {patient.patient_identifier}</p>
                {patient.notes && <p><strong>Notes:</strong> {patient.notes}</p>}
                <p><small>Added on: {new Date(patient.created_at).toLocaleDateString()}</small></p>
                {/* Add actions like 'Mark as Contacted' or 'Remove' if needed */}
              </li>
            ))}
          </ul>
        ) : (
          <p>No patients currently added for follow-up for this camp.</p>
        )}
      </div>
    </div>
  );
};

export default ManageFollowUpsPage;
