// src/components/ConnectWithAuthoritiesPage.jsx
import React, { useState, useEffect, useCallback } from 'react';
import api from '../services/api';
import ChatInterface from './ChatInterface';
import '../App.css';

// ACCEPT campId and setCurrentPage as props from App.jsx
const ConnectWithAuthoritiesPage = ({ campId, currentUser, setCurrentUser, setCurrentPage }) => {

  const [authorities, setAuthorities] = useState([]);
  const [selectedAuthorityId, setSelectedAuthorityId] = useState('');
  const [campConnections, setCampConnections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [activeChat, setActiveChat] = useState(null);
  const [campName, setCampName] = useState('');

  const fetchCampDetails = useCallback(async () => {
    if (!campId) { // Check prop directly
      console.error("Failed to fetch camp details: campId prop is missing.");
      setError(prev => (prev ? prev + ' ' : '') + 'Failed to load camp name (missing camp ID).');
      return;
    }
    try {
      const response = await api.get(`/organizer/camps/${campId}`);
      setCampName(response.data.name);
    } catch (err) {
      console.error("Failed to fetch camp details:", err);
      setError(prev => (prev ? prev + ' ' : '') + 'Failed to load camp name.');
    }
  }, [campId]);

  const fetchAuthorities = useCallback(async () => {
    try {
      const response = await api.get('/local-organisations');
      setAuthorities(response.data || []);
    } catch (err) {
      setError('Failed to fetch local authorities. ' + (err.response?.data?.error || err.message));
    }
  }, []);

  const fetchCampConnections = useCallback(async () => {
    if (!campId) { // Check prop directly
        setError('Failed to fetch camp connections (missing camp ID).');
        return;
    }
    try {
      const response = await api.get(`/organizer/camp/${campId}/connections`);
      setCampConnections(response.data || []);
    } catch (err)
{
      setError('Failed to fetch camp connections. ' + (err.response?.data?.error || err.message));
    }
  }, [campId]);

  useEffect(() => {
    if (!currentUser || currentUser.userType !== 'organizer') {
      // navigate('/login'); // REPLACE with setCurrentPage
      setCurrentPage('login');
      return;
    }
    if (!campId) { // Add check for campId prop
        setError("Cannot load page: Camp ID is missing.");
        setLoading(false);
        return;
    }
    setLoading(true);
    Promise.all([fetchAuthorities(), fetchCampConnections(), fetchCampDetails()])
      .finally(() => setLoading(false));
  // Add campId and setCurrentPage to dependencies, remove navigate
  }, [currentUser, campId, setCurrentPage, fetchAuthorities, fetchCampConnections, fetchCampDetails]);

  const handleSendRequest = async (e) => {
    e.preventDefault();
    if (!selectedAuthorityId) {
      setError('Please select an authority to connect with.');
      return;
    }
    setError('');
    setSuccess('');
    try {
      const payload = {
        campId: parseInt(campId),
        localOrgId: parseInt(selectedAuthorityId),
      };
      await api.post('/chat/request', payload);
      setSuccess('Connection request sent successfully!');
      setSelectedAuthorityId('');
      fetchCampConnections();
    } catch (err) {
      setError('Failed to send connection request. ' + (err.response?.data?.error || err.message));
      console.error("Send request error:", err);
    }
  };

  const openChat = (connection) => {
    setActiveChat({
      connectionId: connection.connection_id,
      chatWithName: connection.local_org_name,
      chatWithUserType: 'Local Authority',
      campName: campName,
    });
  };

  if (loading) return <div className="page-loading-message">Loading connection options...</div>;

  if (activeChat) {
    return (
      <>
        {/* Navbar and Footer are part of App.jsx's structure.
            If this component needs them independently, ensure props are passed.
            <Navbar currentUser={currentUser} setCurrentUser={setCurrentUser} setCurrentPage={setCurrentPage} />
        */}
        <div className="container-page page-padding"> {/* Use container-page for consistency */}
          <ChatInterface
            connectionId={activeChat.connectionId}
            currentUser={currentUser}
            chatWithName={activeChat.chatWithName}
            chatWithUserType={activeChat.chatWithUserType}
            campName={activeChat.campName}
            onClose={() => setActiveChat(null)}
          />
        </div>
        {/* <Footer /> */}
      </>
    );
  }

  return (
    <>
      {/* <Navbar currentUser={currentUser} setCurrentUser={setCurrentUser} setCurrentPage={setCurrentPage} /> */}
      <div className="container-page connect-authorities-page"> {/* Use container-page */}
        <button 
            // onClick={() => navigate(`/organizer/camps/${campId}`)} // REPLACE with setCurrentPage
            onClick={() => setCurrentPage('campDetails')} // Navigate back to campDetails
            className="back-button" 
            style={{marginBottom: '20px'}}
        >
            &larr; Back to Camp Details
        </button>
        <div className="page-header">
          <h1>Connect with Local Authorities for Camp: {campName || `ID ${campId}`}</h1>
        </div>

        {error && <div className="error-message">{error}</div>}
        {success && <div className="success-message">{success}</div>}

        <div className="page-section">
          <h2>Send New Connection Request</h2>
          <form onSubmit={handleSendRequest} className="connect-form"> {/* Added className for potential styling */}
            <div className="form-group">
              <label htmlFor="authority-select">Select Authority:</label>
              <select
                id="authority-select"
                value={selectedAuthorityId}
                onChange={(e) => setSelectedAuthorityId(e.target.value)}
                required
              >
                <option value="">-- Choose an Authority --</option>
                {authorities.map(auth => (
                  <option key={auth.id} value={auth.id}>
                    {auth.name} ({auth.address || 'Address not specified'})
                  </option>
                ))}
              </select>
            </div>
            <button type="submit" className="button-primary">Send Request</button>
          </form>
        </div>

        <div className="page-section">
          <h2>Existing Connections for this Camp</h2>
          {campConnections.length > 0 ? (
            <ul className="connections-list">
              {campConnections.map(conn => (
                <li key={conn.connection_id} className="connection-item">
                  <p><strong>Authority:</strong> {conn.local_org_name}</p>
                  <p><strong>Status:</strong> <span className={`status-badge status-${conn.status}`}>{conn.status}</span></p>
                  <p><strong>Requested:</strong> {new Date(conn.requested_at).toLocaleString()}</p>
                  {conn.responded_at && <p><strong>Responded:</strong> {new Date(conn.responded_at).toLocaleString()}</p>}
                  {conn.status === 'accepted' && (
                    <button onClick={() => openChat(conn)} className="button-chat">Chat with Authority</button>
                  )}
                </li>
              ))}
            </ul>
          ) : (
            <p>No existing connections or requests for this camp.</p>
          )}
        </div>
      </div>
      {/* <Footer /> */}
    </>
  );
};

export default ConnectWithAuthoritiesPage;
