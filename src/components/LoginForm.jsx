// src/components/LoginForm.jsx
import React, { useState } from 'react';

const LoginForm = ({ setCurrentPage, onLoginSuccess }) => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await fetch('https://camp-mdxq.onrender.com/api/login', { // Ensure your backend URL is correct
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (response.ok) {
        // Call a success handler passed via props, e.g., to store user data and token
        if (onLoginSuccess) onLoginSuccess(data.user);
        // Redirect to home or dashboard
        // setCurrentPage('home'); // Or a dashboard page
      } else {
        setError(data.error || `Login failed (Status: ${response.status})`);
      }
    } catch (err) {
      console.error("Login error:", err);
      setError("An unexpected error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-form-container">
      <h2>Login to Your Account</h2>
      <form onSubmit={handleSubmit} className="auth-form">
        {error && <p className="error-message">{error}</p>}

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
          <label htmlFor="password">Password:</label>
          <input
            type="password"
            id="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            required
            placeholder="Enter your password"
          />
        </div>

        <button type="submit" disabled={loading} className="auth-button">
          {loading ? 'Logging In...' : 'Login'}
        </button>
        <p className="switch-form-text">
          Don't have an account?{' '}
          <button type="button" onClick={() => setCurrentPage('signup')} className="link-button">
            Sign up here
          </button>
        </p>
      </form>
    </div>
  );
};

export default LoginForm;
