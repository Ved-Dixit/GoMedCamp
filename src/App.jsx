// src/App.jsx
import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Home from './components/Home';
import About from './components/About';
import Footer from './components/Footer';
import SignUpForm from './components/SignUpForm';
import LoginForm from './components/LoginForm';
import OrganizerDashboard from './components/OrganizerDashboard';
import CampDetailsPage from './components/CampDetailsPage';
import CampResourceManagementPage from './components/CampResourceManagementPage';
import AuthorityDashboard from './components/AuthorityDashboard';
import ConnectWithAuthoritiesPage from './components/ConnectWithAuthoritiesPage';

// Imports for Patient and related Organizer features
import PatientDashboard from './components/PatientDashboard';
import ViewCampReviewsPage from './components/ViewCampReviewsPage'; // For viewing camp reviews
import ManageFollowUpsPage from './components/ManageFollowUpsPage'; // For managing patient follow-ups

import api from './services/api'; // Ensure api.js is correctly set up
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'light');
  const [currentUser, setCurrentUser] = useState(JSON.parse(localStorage.getItem('gomedcampUser')) || null);
  const [selectedCampId, setSelectedCampId] = useState(JSON.parse(localStorage.getItem('gomedcampSelectedCampId')) || null);

  useEffect(() => {
    document.body.className = '';
    document.body.classList.add(`${theme}-theme`);
    localStorage.setItem('theme', theme);
  }, [theme]);

  useEffect(() => {
    if (currentUser) {
      localStorage.setItem('gomedcampUser', JSON.stringify(currentUser));
      // Set X-User-Id header for all api requests if user is logged in
      if (currentUser.id) { // Ensure currentUser.id exists
        api.defaults.headers.common['X-User-Id'] = currentUser.id.toString();
      }
    } else {
      localStorage.removeItem('gomedcampUser');
      // Remove X-User-Id header if user logs out
      delete api.defaults.headers.common['X-User-Id'];
    }
  }, [currentUser]);

  useEffect(() => {
    if (selectedCampId) {
      localStorage.setItem('gomedcampSelectedCampId', JSON.stringify(selectedCampId));
    } else {
      localStorage.removeItem('gomedcampSelectedCampId');
    }
  }, [selectedCampId]);


  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  const handleLoginSuccess = (userData) => {
    console.log("Login successful, user data from backend:", userData);
    setCurrentUser(userData);
    if (userData.userType === 'organizer') {
      setCurrentPage('organizerDashboard');
    } else if (userData.userType === 'local_organisation') {
      setCurrentPage('authorityDashboard');
    } else if (userData.userType === 'requester') { // Patients are 'requester' type
      setCurrentPage('patientDashboard');
    }
     else {
      // Fallback for any other userType or if userType is missing
      setCurrentPage('home');
    }
  };

  const handleSignUpSuccess = (userData) => {
    console.log("Sign up successful:", userData);
    // For all user types, redirect to login after signup.
    // Login will then handle role-based dashboard redirection.
    setCurrentPage('login');
  };

  const handleLogout = () => {
    setCurrentUser(null);
    setSelectedCampId(null);
    setCurrentPage('home');
  };

  const navigateToCampDetails = (campId) => {
    console.log(`App.jsx: Navigating to details for camp ID: ${campId}`);
    setSelectedCampId(campId);
    setCurrentPage('campDetails');
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <Home />;
      case 'about':
        return <About />;
      case 'login':
        return <LoginForm setCurrentPage={setCurrentPage} onLoginSuccess={handleLoginSuccess} />;
      case 'signup':
        // SignUpForm creates 'requester' for patients
        return <SignUpForm setCurrentPage={setCurrentPage} onSignUpSuccess={handleSignUpSuccess} />;

      // Organizer Pages
      case 'organizerDashboard':
        if (currentUser && currentUser.userType === 'organizer') {
          return <OrganizerDashboard currentUser={currentUser} navigateToCampDetails={navigateToCampDetails} />;
        }
        setCurrentPage(currentUser ? 'home' : 'login');
        return null;

      case 'campDetails':
        if (currentUser && currentUser.userType === 'organizer' && selectedCampId) {
          return <CampDetailsPage campId={selectedCampId} setCurrentPage={setCurrentPage} currentUser={currentUser} />;
        }
        if (!currentUser) setCurrentPage('login');
        else if (currentUser.userType === 'organizer') setCurrentPage('organizerDashboard');
        else setCurrentPage('home');
        return null;

      case 'campResourceManagement':
        if (currentUser && currentUser.userType === 'organizer' && selectedCampId) {
          return <CampResourceManagementPage campId={selectedCampId} currentUser={currentUser} setCurrentPage={setCurrentPage} />;
        }
        if (!currentUser) setCurrentPage('login');
        else if (currentUser.userType === 'organizer') setCurrentPage('organizerDashboard');
        else setCurrentPage('home');
        return null;

      case 'connectWithAuthorities':
        if (currentUser && currentUser.userType === 'organizer' && selectedCampId) {
          return <ConnectWithAuthoritiesPage campId={selectedCampId} currentUser={currentUser} setCurrentUser={setCurrentUser} setCurrentPage={setCurrentPage} />;
        }
        if (!currentUser) setCurrentPage('login');
        else if (currentUser.userType === 'organizer') {
            console.log("App.jsx: Organizer, but no selectedCampId for connectWithAuthorities. Redirecting to dashboard.");
            setCurrentPage('organizerDashboard');
        } else {
            setCurrentPage('home');
        }
        return null;

      // Authority Dashboard
      case 'authorityDashboard':
        if (currentUser && currentUser.userType === 'local_organisation') {
          return <AuthorityDashboard currentUser={currentUser} setCurrentUser={setCurrentUser} setCurrentPage={setCurrentPage} />;
        }
        setCurrentPage(currentUser ? 'home' : 'login');
        return null;

      // Patient Dashboard
      case 'patientDashboard':
        if (currentUser && currentUser.userType === 'requester') { // Patients are 'requester' type
          return <PatientDashboard currentUser={currentUser} setCurrentPage={setCurrentPage} />;
        }
        setCurrentPage(currentUser ? 'home' : 'login'); // Fallback
        return null;

      // Organizer pages related to camp details (Reviews and Follow-ups)
      case 'viewCampReviews':
        if (currentUser && currentUser.userType === 'organizer' && selectedCampId) {
          return <ViewCampReviewsPage campId={selectedCampId} currentUser={currentUser} setCurrentPage={setCurrentPage} />;
        }
        if (!currentUser) setCurrentPage('login');
        else if (currentUser.userType === 'organizer') {
            console.log("App.jsx: Organizer, but no selectedCampId for viewCampReviews. Redirecting to dashboard.");
            setCurrentPage('organizerDashboard');
        } else {
            setCurrentPage('home');
        }
        return null;

      case 'manageCampFollowUps':
        if (currentUser && currentUser.userType === 'organizer' && selectedCampId) {
          return <ManageFollowUpsPage campId={selectedCampId} currentUser={currentUser} setCurrentPage={setCurrentPage} />;
        }
        if (!currentUser) setCurrentPage('login');
        else if (currentUser.userType === 'organizer') {
            console.log("App.jsx: Organizer, but no selectedCampId for manageCampFollowUps. Redirecting to dashboard.");
            setCurrentPage('organizerDashboard');
        } else {
            setCurrentPage('home');
        }
        return null;

      default:
        setCurrentPage('home');
        return <Home />;
    }
  };

  const isDashboardPage = currentPage === 'organizerDashboard' || currentPage === 'authorityDashboard' || currentPage === 'patientDashboard';

  const mainContentStyle = isDashboardPage
    ? {padding: '20px', maxWidth: '1200px', margin: '0 auto', flexGrow: 1} // Unified dashboard padding
    : {};

  const showFooter = !isDashboardPage; // Hide footer on all dashboards for more space

  return (
    <div className="app-container">
      <Navbar
        currentPage={currentPage}
        setCurrentPage={setCurrentPage}
        theme={theme}
        toggleTheme={toggleTheme}
        currentUser={currentUser}
        onLogout={handleLogout}
      />
      <main className="main-content" style={mainContentStyle}>
        {renderPage()}
      </main>
      {showFooter && <Footer />}
    </div>
  );
}

export default App;
