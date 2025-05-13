// src/components/SignUpForm.jsx
import React, { useState } from 'react';
import api from '../services/api'; // Assuming api.js is set up for network requests

const SignUpForm = ({ setCurrentPage, onSignUpSuccess }) => {
  const [formData, setFormData] = useState({
    name: '', // Frontend state, maps to 'username' for backend
    email: '',
    phoneNumber: '', // Frontend state, maps to 'phone_number' for backend
    password: '',
    confirmPassword: '',
    userType: 'requester', // Default role as per app.py (patient/local leader)
    address: '',
  });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);

  // User types available for signup, matching backend expectations
  // 'requester' is the general user/patient type for signup.
  const userTypes = [
    { value: 'requester', label: 'General User / Patient' },
    { value: 'organizer', label: 'Camp Organizer' },
    { value: 'local_organisation', label: 'Local Medical Body/Organisation' },
  ];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    // Clear address if userType is not local_organisation
    if (name === 'userType' && value !== 'local_organisation') {
      setFormData(prev => ({ ...prev, address: '' }));
    }
    setError(''); // Clear error on change
    setSuccess(''); // Clear success on change
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    if (formData.userType === 'local_organisation' && !formData.address.trim()) {
      setError("Address is required for Local Medical Body/Organisation.");
      return;
    }

    setLoading(true);

    // Prepare payload according to backend expectations
    const payload = {
      username: formData.name, // Backend expects 'username'
      email: formData.email,
      phone_number: formData.phoneNumber, // Backend expects 'phone_number'
      password: formData.password,
      userType: formData.userType, // Backend expects 'userType' (camelCase)
      address: formData.userType === 'local_organisation' ? formData.address.trim() : null,
    };

    try {
      // Using api.js service for consistency
      const response = await api.post('/signup', payload);

      setSuccess(response.data.message || "Sign up successful! You can now log in.");
      if (onSignUpSuccess) {
        onSignUpSuccess(response.data.user); // Pass user data to parent if needed
      }
      // Reset form
      setFormData({
          name: '', email: '', phoneNumber: '', password: '',
          confirmPassword: '', userType: 'requester', address: ''
      });

    } catch (err) {
      console.error("Sign up error:", err);
      setError(err.response?.data?.error || err.message || "An unexpected error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-form-container">
      <h2>Create Account</h2>
      <form onSubmit={handleSubmit} className="auth-form">
        {error && <p className="error-message">{error}</p>}
        {success && <p className="success-message">{success}</p>}

        <div className="form-group">
          <label htmlFor="userType">I am a:</label>
          <select
            id="userType"
            name="userType"
            value={formData.userType}
            onChange={handleChange}
            required
          >
            {userTypes.map(type => (
              <option key={type.value} value={type.value}>{type.label}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="name">Full Name (Username):</label>
          <input
            type="text"
            id="name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            required
            placeholder="Enter your full name"
          />
        </div>

        <div className="form-group">
          <label htmlFor="email">Email:</label>
          <input
            type="email"
            id="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            required
            placeholder="Enter your email address"
          />
        </div>

        <div className="form-group">
          <label htmlFor="phoneNumber">Phone Number:</label>
          <input
            type="tel"
            id="phoneNumber"
            name="phoneNumber"
            value={formData.phoneNumber}
            onChange={handleChange}
            required
            placeholder="Enter your phone number"
          />
        </div>

        {formData.userType === 'local_organisation' && (
          <div className="form-group">
            <label htmlFor="address">Address (for Local Medical Body):</label>
            <textarea
              id="address"
              name="address"
              value={formData.address}
              onChange={handleChange}
              required={formData.userType === 'local_organisation'}
              placeholder="Enter full address"
              rows="3"
            />
          </div>
        )}

        <div className="form-group">
          <label htmlFor="password">Password:</label>
          <input
            type="password"
            id="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            required
            placeholder="Create a password (min. 6 characters)"
            minLength="6"
          />
        </div>

        <div className="form-group">
          <label htmlFor="confirmPassword">Confirm Password:</label>
          <input
            type="password"
            id="confirmPassword"
            name="confirmPassword"
            value={formData.confirmPassword}
            onChange={handleChange}
            required
            placeholder="Confirm your password"
            minLength="6"
          />
        </div>

        <button type="submit" disabled={loading} className="auth-button">
          {loading ? 'Signing Up...' : 'Sign Up'}
        </button>
        <p className="switch-form-text">
          Already have an account?{' '}
          <button type="button" onClick={() => setCurrentPage('login')} className="link-button">
            Login here
          </button>
        </p>
      </form>
    </div>
  );
};

export default SignUpForm;
