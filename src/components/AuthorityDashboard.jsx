// src/components/AuthorityDashboard.jsx
import React, { useState, useEffect, useCallback } from 'react';
import api from '../services/api';
import ChatInterface from './ChatInterface';
import '../App.css';

// ACCEPT setCurrentPage as a prop from App.jsx
const AuthorityDashboard = ({ currentUser, setCurrentUser, setCurrentPage }) => {
  const [pendingRequests, setPendingRequests] = useState([]);
  const [acceptedConnections, setAcceptedConnections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeChat, setActiveChat] = useState(null);

  const fetchPendingRequests = useCallback(async () => {
    if (!currentUser || currentUser.userType !== 'local_organisation') return;
    try {
      const response = await api.get(`/local-organisation/${currentUser.id}/requests`); // Fetches pending
      setPendingRequests(response.data.pendingRequests || []);
    } catch (err) {
      setError('Failed to fetch pending requests. ' + (err.response?.data?.error || err.message));
      console.error("Fetch pending requests error:", err);
    }
  }, [currentUser]);

  const fetchAcceptedConnections = useCallback(async () => {
    if (!currentUser || currentUser.userType !== 'local_organisation') return;
    try {
      const response = await api.get(`/local-organisation/${currentUser.id}/connections?status=accepted`);
      setAcceptedConnections(response.data || []);
    } catch (err) {
      setError('Failed to fetch accepted connections. ' + (err.response?.data?.error || err.message));
      console.error("Fetch accepted connections error:", err);
    }
  }, [currentUser]);

  useEffect(() => {
    if (!currentUser || currentUser.userType !== 'local_organisation') {
      setCurrentPage('login'); // Use setCurrentPage for navigation
      return;
    }
    setLoading(true);
    Promise.all([fetchPendingRequests(), fetchAcceptedConnections()])
      .finally(() => setLoading(false));
  // Add setCurrentPage to dependencies, remove navigate
  }, [currentUser, setCurrentPage, fetchPendingRequests, fetchAcceptedConnections]);

  const handleRequestAction = async (requestId, action) => {
    try {
      await api.put(`/chat/request/${requestId}/respond`, { status: action });
      // Refresh both lists
      fetchPendingRequests();
      fetchAcceptedConnections();
      setError(''); // Clear previous errors
    } catch (err) {
      setError(`Failed to ${action} request. ` + (err.response?.data?.error || err.message));
      console.error("Request action error:", err);
    }
  };

  const openChat = (connection) => {
    setActiveChat({
      connectionId: connection.connection_id,
      chatWithName: connection.organizer_name,
      chatWithUserType: 'Organizer', // Authority chats with Organizer
      campName: connection.camp_name,
    });
  };

  if (loading) return <div className="page-loading-message">Loading Authority Dashboard...</div>;

  if (activeChat) {
    return (
      <>
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
      <div className="container-page authority-dashboard-page"> {/* Use container-page */}
        <div className="dashboard-header">
          <h1>Authority Dashboard</h1>
          <p>Welcome, {currentUser?.username}! Manage camp connection requests and communications.</p>
        </div>

        {error && <div className="error-message">{error}</div>}

        <div className="dashboard-section">
          <h2>Pending Connection Requests</h2>
          {pendingRequests.length > 0 ? (
            <ul className="requests-list">
              {pendingRequests.map(req => (
                <li key={req.request_id} className="request-item">
                  <p><strong>Camp Name:</strong> {req.camp_name}</p>
                  <p><strong>Organizer:</strong> {req.organizer_name}</p>
                  <p><strong>Requested:</strong> {new Date(req.requested_at).toLocaleString()}</p>
                  <p><strong>Camp Starts:</strong> {new Date(req.camp_start_date).toLocaleDateString()}</p>
                  <div className="request-actions">
                    <button onClick={() => handleRequestAction(req.request_id, 'accepted')} className="button-accept">Accept</button>
                    <button onClick={() => handleRequestAction(req.request_id, 'declined')} className="button-reject">Decline</button>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p>No pending requests at the moment.</p>
          )}
        </div>

        <div className="dashboard-section">
          <h2>Active Connections & Chats</h2>
          {acceptedConnections.length > 0 ? (
            <ul className="connections-list">
              {acceptedConnections.map(conn => (
                <li key={conn.connection_id} className="connection-item">
                  <p><strong>Camp Name:</strong> {conn.camp_name}</p>
                  <p><strong>Organizer:</strong> {conn.organizer_name}</p>
                  <p><strong>Status:</strong> <span className={`status-badge status-${conn.status}`}>{conn.status}</span></p>
                  <p><strong>Connected Since:</strong> {new Date(conn.responded_at || conn.requested_at).toLocaleString()}</p>
                  {conn.status === 'accepted' && (
                    <button onClick={() => openChat(conn)} className="button-chat">Chat with Organizer</button>
                  )}
                </li>
              ))}
            </ul>
          ) : (
            <p>No active connections to display.</p>
          )}
        </div>
      </div>
      {/* <Footer /> */}
    </>
  );
};

export default AuthorityDashboard;
