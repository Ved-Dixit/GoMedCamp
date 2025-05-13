// src/components/Navbar.jsx
import React from 'react';
import ThemeToggle from './ThemeToggle';
import logoLight from '../assets/logo.png';

const Navbar = ({ currentPage, setCurrentPage, theme, toggleTheme, currentUser, onLogout }) => {
  return (
    <nav className="navbar">
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <img
          src={logoLight}
          alt="GoMedCamp Logo"
          className="logo"
          onClick={() => setCurrentPage('home')}
          style={{cursor: 'pointer', height: '40px', marginRight: '15px'}}
        />
        <div className="nav-links">
          <a
            href="#home"
            className={currentPage === 'home' ? 'active' : ''}
            onClick={(e) => { e.preventDefault(); setCurrentPage('home'); }}
          >
            Home
          </a>
          <a
            href="#about"
            className={currentPage === 'about' ? 'active' : ''}
            onClick={(e) => { e.preventDefault(); setCurrentPage('about'); }}
          >
            About
          </a>
          {currentUser && currentUser.userType === 'organizer' && (
            <a
              href="#organizer-dashboard"
              className={currentPage === 'organizerDashboard' ? 'active' : ''}
              onClick={(e) => { e.preventDefault(); setCurrentPage('organizerDashboard'); }}
            >
              Organizer Dashboard
            </a>
          )}
          {currentUser && currentUser.userType === 'local_organisation' && (
            <a
              href="#authority-dashboard"
              className={currentPage === 'authorityDashboard' ? 'active' : ''}
              onClick={(e) => { e.preventDefault(); setCurrentPage('authorityDashboard'); }}
            >
              Authority Dashboard
            </a>
          )}
          {/* Updated to check for 'requester' userType for Patient Dashboard link */}
          {/* This aligns with SignUpForm creating 'requester' for patients, */}
          {/* and PatientDashboard component expecting 'requester'. */}
          {currentUser && currentUser.userType === 'requester' && (
            <a
              href="#patient-dashboard"
              className={currentPage === 'patientDashboard' ? 'active' : ''}
              onClick={(e) => { e.preventDefault(); setCurrentPage('patientDashboard'); }}
            >
              Patient Dashboard
            </a>
          )}
        </div>
      </div>
      <div className="nav-actions" style={{ display: 'flex', alignItems: 'center' }}>
        {currentUser ? (
          <>
            <span style={{ marginRight: '1rem', fontWeight: '500', color: 'var(--navbar-text-color, inherit)' }}>
              Hi, {currentUser.username || currentUser.email}!
              <span style={{fontSize: '0.8em', opacity: '0.8'}}> ({currentUser.userType.replace(/_/g, ' ')})</span>
            </span>
            <button onClick={onLogout} className="auth-button logout-button">Logout</button>
          </>
        ) : (
          <>
            <button onClick={() => setCurrentPage('login')} className="auth-button login-button">Login</button>
            <button onClick={() => setCurrentPage('signup')} className="auth-button signup-button">Sign Up</button>
          </>
        )}
        <ThemeToggle theme={theme} toggleTheme={toggleTheme} className="theme-toggle-button" />
      </div>
    </nav>
  );
};

export default Navbar;
