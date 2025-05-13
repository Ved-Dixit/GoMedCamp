// src/components/CampDetailsPage.jsx
import React, { useEffect, useState } from 'react';
import api from '../services/api';
import '../App.css'; // Assuming common styles are here

// Sample data for government schemes (can be moved to a separate file if large)
const governmentSchemesData = [
  { id: 1, name: "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana (AB PM-JAY)", link: "https://pmjay.gov.in/" },
  { id: 2, name: "National Health Mission (NHM)", link: "https://nhm.gov.in/" },
  { id: 3, name: "Janani Shishu Suraksha Karyakram (JSSK)", link: "https://nhm.gov.in/index1.php?lang=1&level=2&sublinkid=842&lid=220" },
  { id: 4, name: "Rashtriya Bal Swasthya Karyakram (RBSK)", link: "https://rbsk.nhm.gov.in/" },
  { id: 5, name: "Pradhan Mantri Surakshit Matritva Abhiyan (PMSMA)", link: "https://pmsma.nhm.gov.in/" },
  { id: 6, name: "National Viral Hepatitis Control Program (NVHCP)", link: "https://nvhcp.gov.in/" },
  { id: 7, name: "Pradhan Mantri Bhartiya Janaushadhi Pariyojana (PMBJP)", link: "http://janaushadhi.gov.in/" },
  { id: 8, name: "National Tobacco Control Programme (NTCP)", link: "https://ntcp.nhp.gov.in/" },
  { id: 9, name: "eSanjeevani - National Teleconsultation Service", link: "https://esanjeevaniopd.in/" }
];


const CampDetailsPage = ({ campId, setCurrentPage, currentUser }) => {
  const [campDetails, setCampDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showSchemesModal, setShowSchemesModal] = useState(false);

  useEffect(() => {
    const fetchCampDetails = async () => {
      if (!campId || !currentUser?.id) {
        setError("Camp ID or User ID is missing for fetching details.");
        setLoading(false);
        return;
      }
      setLoading(true);
      setError('');
      try {
        // Using fetch as per the provided context file
        const response = await fetch(`https://camp-mdxq.onrender.com/api/organizer/camps/${campId}`, {
          headers: {
            'Content-Type': 'application/json',
            'X-User-Id': currentUser.id.toString(), // Ensure X-User-Id is sent
          },
        });
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ error: "Failed to parse error response" }));
          throw new Error(errorData.error || `Failed to fetch camp details (Status: ${response.status})`);
        }
        const data = await response.json();
        setCampDetails(data);
      } catch (err) {
        console.error("Error fetching camp details:", err);
        setError(err.message || "Could not load camp details.");
        // Fallback to ensure campDetails is not null if an error occurs, showing at least the ID
        if (!campDetails) { // Only set this if campDetails is still null
            setCampDetails({ id: campId, name: `Camp #${campId} (Details partially loaded)` });
        }
      } finally {
        setLoading(false);
      }
    };

    fetchCampDetails();
  }, [campId, currentUser]); // Removed campDetails from dependency array to prevent re-fetch loops on setCampDetails

  const handleFeatureClick = (featureName) => {
    if (featureName === 'Staff and Medical Equipment Management') {
      setCurrentPage('campResourceManagement');
    } else if (featureName === 'Government Schemes') {
      setShowSchemesModal(true);
    } else if (featureName === 'Connecting with Local Authorities') {
      setCurrentPage('connectWithAuthorities');
    } else if (featureName === 'Managing Reviews') {
      setCurrentPage('viewCampReviews'); 
    } else if (featureName === 'Patient Follow-up & Referrals') {
      setCurrentPage('manageCampFollowUps');
    } else {
      alert(`Feature clicked: ${featureName}. This functionality may be in development or coming soon.`);
    }
  };

  if (loading) return <p className="page-loading-message">Loading camp details...</p>;
  
  // If there's an error and campDetails couldn't even be minimally set (e.g., no campId)
  if (error && !campDetails) return <p className="error-message" style={{textAlign: 'center', padding: '20px'}}>{error}</p>;
  
  // If campDetails is null after loading (shouldn't happen if fallback in catch works, but good check)
  if (!campDetails) return <p className="info-message" style={{textAlign: 'center', padding: '20px'}}>Camp not found or details could not be loaded.</p>;

  const displayCampName = campDetails.name || `Camp #${campDetails.id}`;

  return (
    <div className="camp-details-container">
      <button 
        onClick={() => setCurrentPage('organizerDashboard')} 
        className="back-button" // This class can be styled for a neutral back button
        style={{
            marginBottom: '20px', 
            backgroundColor: 'var(--secondary-button-bg)', 
            color: 'var(--secondary-button-text)', 
            border: '1px solid var(--secondary-button-border)'
        }}
      >
        &larr; Back to Organizer Dashboard
      </button>
      <div className="camp-details-welcome-panel">
        <h2>Welcome, Organizer!</h2>
        <p>
          Thank you for your dedication and hard work in organizing <strong>{displayCampName}</strong>.
          Your efforts in bringing healthcare services to the community are invaluable.
          This platform is designed to support you in managing your camp effectively
          and making a significant impact on public health. We appreciate your commitment
          to making a difference!
        </p>
        {/* Display error here if campDetails were partially loaded but an error still occurred */}
        {error && campDetails.name && <p className="error-message" style={{fontSize: '0.9em', marginTop: '10px'}}>Note: {error}</p>}
      </div>
      <div className="camp-details-features-panel">
        <h3>Camp Management Features for "{displayCampName}"</h3>
        <button onClick={() => handleFeatureClick('Staff and Medical Equipment Management')}>
          Staff & Medical Equipment Management
        </button>
        <button onClick={() => handleFeatureClick('Managing Reviews')}>
          Managing Reviews
        </button>
        <button onClick={() => handleFeatureClick('Patient Follow-up & Referrals')}>
          Patient Follow-up & Referrals
        </button>
        <button onClick={() => handleFeatureClick('Connecting with Local Authorities')}>
          Connecting with Local Authorities
        </button>
        <button onClick={() => handleFeatureClick('Government Schemes')}>
          Government Schemes
        </button>
      </div>

      {showSchemesModal && (
        <div className="modal-overlay" onClick={() => setShowSchemesModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Latest Government Health Schemes</h3>
              <button onClick={() => setShowSchemesModal(false)} className="modal-close-button">&times;</button>
            </div>
            <ul className="schemes-list">
              {governmentSchemesData.map(scheme => (
                <li key={scheme.id}>
                  <a href={scheme.link} target="_blank" rel="noopener noreferrer">
                    {scheme.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default CampDetailsPage;
